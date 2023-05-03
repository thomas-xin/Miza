# Smudge invaded, on the 1st August 2020 this became my territory *places flag* B3

#!/usr/bin/python3

import common
from common import *
import h2

ADDRESS = AUTH.get("webserver_address") or "0.0.0.0"
if ADDRESS == "0.0.0.0":
    ADDRESS = "127.0.0.1"

create_future_ex(get_colour_list)
create_future_ex(load_emojis)
create_future_ex(load_timezones)

# Allows importing from commands and misc directories.
sys.path.insert(1, "commands")
sys.path.insert(1, "misc")

heartbeat_proc = psutil.Popen([python, "misc/heartbeat.py"])


# Main class containing all global bot data.
class Bot(discord.Client, contextlib.AbstractContextManager, collections.abc.Callable):

    github = AUTH.get("github") or "https://github.com/thomas-xin/Miza"
    rcc_invite = AUTH.get("rcc_invite") or "https://discord.gg/cbKQKAr"
    discord_icon = "https://cdn.discordapp.com/embed/avatars/0.png"
    twitch_url = "https://www.twitch.tv/-"
    webserver = AUTH.get("webserver") or "https://mizabot.xyz"
    kofi_url = AUTH.get("kofi_url") or "https://ko-fi.com/mizabot"
    rapidapi_url = AUTH.get("rapidapi_url") or "https://rapidapi.com/thomas-xin/api/miza"
    raw_webserver = webserver
    server_init = False
    heartbeat = "heartbeat.tmp"
    heartbeat_ack = "heartbeat_ack.tmp"
    restart = "restart.tmp"
    shutdown = "shutdown.tmp"
    activity = 0
    caches = ("guilds", "channels", "users", "roles", "emojis", "messages", "members", "attachments", "deleted", "banned", "colours")
    statuses = (discord.Status.online, discord.Status.idle, discord.Status.dnd, discord.Streaming, discord.Status.invisible)
    # Default command prefix
    prefix = AUTH.get("prefix", "~")
    # This is a fixed ID apparently
    deleted_user = 456226577798135808
    _globals = globals()
    intents = discord.Intents(
        guilds=True,
        members=True,
        bans=True,
        emojis=True,
        webhooks=True,
        voice_states=True,
        presences=False,
        messages=True,
        reactions=True,
        typing=True,
    )
    intents.value |= 32768 # message content intent because discord dumb
    allowed_mentions = discord.AllowedMentions(
        everyone=False,
        users=True,
        roles=False,
        replied_user=False,
    )
    connect_ready = concurrent.futures.Future()
    socket_responses = deque(maxlen=256)
    try:
        shards = int(sys.argv[1])
    except IndexError:
        shards = 2
    premium_server = 247184721262411776
    premium_roles = {
        1052645637033824346: 1,
        1052645761638215761: 2,
        1052647823188967444: 3,
    }
    active_categories = set(AUTH.setdefault("active_categories", ["MAIN", "STRING", "ADMIN", "VOICE", "IMAGE", "WEBHOOK", "FUN"]))

    def __init__(self, cache_size=65536, timeout=24):
        # Initializes client (first in __mro__ of class inheritance)
        self.start_time = utc()
        self.monkey_patch()
        super().__init__(
            loop=eloop,
            max_messages=256,
            heartbeat_timeout=64,
            chunk_guilds_at_startup=False,
            guild_ready_timeout=8,
            intents=self.intents,
            allowed_mentions=self.allowed_mentions,
            assume_unsync_clock=True,
        )
        self.cache_size = cache_size
        # Base cache: contains all other caches
        self.cache = fcdict((c, fdict()) for c in self.caches)
        self.timeout = timeout
        self.set_classes()
        self.set_client_events()
        self.bot = self
        self.client = super()
        self.closed = False
        self.loaded = False
        # Channel-Webhook cache: for accessing all webhooks for a channel.
        self.cw_cache = cdict()
        self.usernames = {}
        self.events = mdict()
        self.react_sem = cdict()
        self.mention = ()
        self.user_loader = set()
        self.users_updated = True
        self.semaphore = Semaphore(2, 1)
        self.ready_semaphore = Semaphore(1, inf)
        self.guild_semaphore = Semaphore(5, inf, rate_limit=5)
        self.load_semaphore = Semaphore(5, inf, rate_limit=1)
        self.user_semaphore = Semaphore(64, inf, rate_limit=8)
        self.disk_semaphore = Semaphore(1, 1, rate_limit=1)
        self.cache_semaphore = Semaphore(1, 1, rate_limit=30)
        self.command_semaphore = Semaphore(262144, 16384)
        self.disk = 0
        self.storage_ratio = 0
        self.file_count = 0
        print("Time:", datetime.datetime.now())
        print("Initializing...")
        # O(1) time complexity for searching directory
        directory = frozenset(os.listdir())
        [os.mkdir(folder) for folder in ("cache", "saves") if folder not in directory]
        if not os.path.exists("saves/filehost"):
            os.mkdir("saves/filehost")
        try:
            self.token = AUTH["discord_token"]
        except KeyError:
            print("ERROR: discord_token not found. Unable to login.")
            self.setshutdown(force=True)
        try:
            owner_id = AUTH["owner_id"]
            if type(owner_id) not in (list, tuple):
                owner_id = (owner_id,)
            self.owners = alist(int(i) for i in owner_id)
        except KeyError:
            self.owners = alist()
            print("WARNING: owner_id not found. Unable to locate owner.")
        # Initialize rest of bot variables
        self.proc = PROC
        self.guild_count = 0
        self.updated = False
        self.started = False
        self.bot_ready = False
        self.ready = False
        self.initialisation_complete = False
        self.stat_timer = 0
        self.last_check = 0
        self.status_iter = xrand(4)
        self.curr_state = azero(4)
        self.ip = None
        self.server = None
        self.audio = None
        self.embed_senders = cdict()
        # Assign bot cache to global variables for convenience
        globals().update(self.cache)
        globals()["messages"] = self.messages = self.MessageCache()
        self.get_modules()
        self.heartbeat_proc = heartbeat_proc

    __str__ = lambda self: str(self.user)
    __repr__ = lambda self: repr(self.user)
    __call__ = lambda self: self
    __exit__ = lambda self, *args, **kwargs: self.close()

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            pass
        this = self._connection
        try:
            return getattr(this, key)
        except AttributeError:
            pass
        this = self.user
        try:
            return getattr(this, key)
        except AttributeError:
            pass
        this = self.__getattribute__("proc")
        return getattr(this, key)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(dir(self._connection))
        data.update(dir(self.user))
        data.update(dir(self.proc))
        return data

    # Waits an amount of seconds and shuts down.
    def setshutdown(self, delay=None, force=False):
        if delay:
            time.sleep(delay)
        if force:
            touch(self.shutdown)
        # force_kill(self.proc)

    def command_options(self, usage, compare=False):
        # default = False
        out = deque()
        for i in usage.split():
            with tracebacksuppressor:
                arg = dict(type=3, name=i, description=i)
                if i.endswith("?"):
                    arg["description"] = "[optional] " + arg["description"][:-1]
                elif i.endswith("*"):
                    arg["description"] = "[zero or more] " + arg["description"][:-1]
                else:
                    if i.endswith("+"):
                        arg["description"] = "[one or more] " + arg["description"][:-1]
                    arg["required"] = True
                    # if not default and usage.count(" "):
                    #     arg["default"] = default = True
                arg["description"] = lim_str(arg["description"], 100)
                if i.startswith("("):
                    arg["name"] = "option"
                    s = i[1:i.rindex(")")]
                    if s.count("|") < 10:
                        arg["choices"] = [dict(name=opt, value=opt) for opt in s.split("|")]
                elif i.startswith("["):
                    arg["name"] = "option"
                elif i.startswith("<"):
                    name = i[1:].split(":", 1)[-1].rsplit(">", 1)[0]
                    if "{" in name:
                        if name.count("{") > 1:
                            if name.count("|") >= 10:
                                arg["name"] = name.split("{", 1)[0]
                            else:
                                for n in name.split("|"):
                                    arg = dict(arg)
                                    arg["name"], flag = n.split("{", 1)
                                    arg["choices"] = [dict(name="true", value=flag.rsplit("}", 1)[0]), dict(name="false", value=None if compare else "")]
                                    out.append(arg)
                                continue
                        else:
                            arg["name"], flag = name.split("{", 1)
                            arg["choices"] = [dict(name="true", value=flag.rsplit("}", 1)[0]), dict(name="false", value=None if compare else "")]
                    else:
                        if name.startswith("("):
                            name = name.strip("()")
                        arg["name"] = name.split("(", 1)[0].replace("|", "-")
                out.append(arg)
        return sorted(out, key=lambda arg: bool(arg.get("required")), reverse=True)

    @tracebacksuppressor
    def create_command(self, data):
        for i in range(16):
            resp = reqs.next().post(
                f"https://discord.com/api/{api}/applications/{self.id}/commands",
                headers={"Content-Type": "application/json", "Authorization": "Bot " + self.token},
                data=orjson.dumps(data),
            )
            self.activity += 1
            if resp.status_code == 429:
                time.sleep(20)
                continue
            if resp.status_code not in range(200, 400):
                print("\n", data, " ", ConnectionError(f"Error {resp.status_code}", resp.text), "\n", sep="")
            print(resp.text)
            return

    def update_slash_commands(self):
        if not AUTH.get("slash_commands"):
            return
        print("Updating global slash commands...")
        with tracebacksuppressor:
            resp = reqs.next().get(
                f"https://discord.com/api/{api}/applications/{self.id}/commands",
                headers=dict(Authorization="Bot " + self.token),
            )
            self.activity += 1
            if resp.status_code not in range(200, 400):
                raise ConnectionError(f"Error {resp.status_code}", resp.text)
            commands = dict((int(c["id"]), c) for c in resp.json() if str(c.get("application_id")) == str(self.id))
            print(f"Successfully loaded {len(commands)} application command{'s' if len(commands) != 1 else ''}.")
        sem = Semaphore(5, inf, 5)
        for catg in self.categories.values():
            for command in catg:
                with tracebacksuppressor:
                    if getattr(command, "msgcmd", None):
                        with sem:
                            aliases = command.msgcmd if type(command.msgcmd) is tuple else (command.parse_name(),)
                            for name in aliases:
                                command_data = dict(name=name, type=3)
                                found = False
                                for i, curr in list(commands.items()):
                                    if curr["name"] == name and curr["type"] == command_data["type"]:
                                        found = True
                                        commands.pop(i)
                                        break
                                if not found:
                                    print(f"creating new message command {command_data['name']}...")
                                    print(command_data)
                                    create_future_ex(self.create_command, command_data, priority=True)
                    if getattr(command, "usercmd", None):
                        with sem:
                            aliases = command.usercmd if type(command.usercmd) is tuple else (command.parse_name(),)
                            for name in aliases:
                                command_data = dict(name=name, type=2)
                                found = False
                                for i, curr in list(commands.items()):
                                    if curr["name"] == name and curr["type"] == command_data["type"]:
                                        found = True
                                        commands.pop(i)
                                        break
                                if not found:
                                    print(f"creating new user command {command_data['name']}...")
                                    print(command_data)
                                    create_future_ex(self.create_command, command_data, priority=True)
                    if getattr(command, "slash", None):
                        with sem:
                            aliases = command.slash if type(command.slash) is tuple else (command.parse_name(),)
                            for name in (full_prune(i) for i in aliases):
                                description = lim_str(command.parse_description(), 100)
                                options = self.command_options(command.usage)
                                command_data = dict(name=name, description=description, type=1)
                                if options:
                                    command_data["options"] = options
                                found = False
                                for i, curr in list(commands.items()):
                                    if curr["name"] == name and curr["type"] == command_data["type"]:
                                        compare = self.command_options(command.usage)
                                        if curr["description"] != description or (compare and curr["options"] != compare or not compare and curr.get("options")):
                                            print(curr)
                                            print(f"{curr['name']}'s slash command does not match, removing...")
                                            for i in range(16):
                                                resp = reqs.next().delete(
                                                    f"https://discord.com/api/{api}/applications/{self.id}/commands/{curr['id']}",
                                                    headers=dict(Authorization="Bot " + self.token),
                                                )
                                                self.activity += 1
                                                if resp.status_code == 429:
                                                    time.sleep(1)
                                                    continue
                                                if resp.status_code not in range(200, 400):
                                                    raise ConnectionError(f"Error {resp.status_code}", resp.text)
                                                break
                                        else:
                                            # print(f"{curr['name']}'s slash command matches, ignoring...")
                                            found = True
                                        commands.pop(i)
                                        break
                                if not found:
                                    print(f"creating new slash command {command_data['name']}...")
                                    print(command_data)
                                    create_future_ex(self.create_command, command_data)
        time.sleep(3)
        for curr in commands.values():
            with tracebacksuppressor:
                print(curr)
                print(f"{curr['name']}'s application command does not exist, removing...")
                resp = reqs.next().delete(
                    f"https://discord.com/api/{api}/applications/{self.id}/commands/{curr['id']}",
                    headers=dict(Authorization="Bot " + self.token),
                )
                self.activity += 1
                if resp.status_code not in range(200, 400):
                    raise ConnectionError(f"Error {resp.status_code}", resp.text)

    async def create_main_website(self, first=False):
        self.start_webserver()
        if first:
            create_thread(webserver_communicate, self)
            print("Generating command json...")
            j = {}
            for category in ("MAIN", "STRING", "ADMIN", "VOICE", "IMAGE", "FUN", "OWNER", "NSFW", "MISC"):
                k = j[category] = {}
                if category not in self.categories:
                    continue
                for command in self.categories[category]:
                    c = k[command.parse_name()] = dict(
                        aliases=[n.strip("_") for n in command.alias],
                        description=command.parse_description(),
                        usage=command.usage,
                        level=str(command.min_level),
                        rate_limit=str(command.rate_limit),
                        example=getattr(command, "example", []),
                        timeout=str(getattr(command, "_timeout_", 1) * self.timeout),
                    )
                    for attr in ("flags", "server_only", "slash"):
                        with suppress(AttributeError):
                            c[attr] = command.attr
            with open("misc/web/static/HELP.json", "w", encoding="utf-8") as f:
                json.dump(j, f, indent="\t")

    def start_webserver(self):
        if self.server and is_strict_running(self.server):
            with suppress():
                force_kill(self.server)
        if os.path.exists("misc/x-server.py") and PORT:
            print("Starting webserver...")
            self.server = psutil.Popen([python, "x-server.py"], cwd=os.getcwd() + "/misc", stderr=subprocess.PIPE, bufsize=65536)
        else:
            self.server = None

    def start_audio_client(self):
        if self.audio:
            with suppress():
                self.audio.kill()
        if os.path.exists("misc/x-audio.py"):
            print("Starting audio client...")
            self.audio = AudioClientInterface()
        else:
            self.audio = None

    # Starts up client.
    def run(self):
        print(f"Logging in...")
        self.audio_client_start = create_future(self.start_audio_client, priority=True)
        with closing(get_event_loop()):
            with tracebacksuppressor:
                get_event_loop().run_until_complete(self.start(self.token))
            with tracebacksuppressor:
                get_event_loop().run_until_complete(self.close())
        self.setshutdown()

    # A reimplementation of the print builtin function.
    def print(self, *args, sep=" ", end="\n"):
        sys.__stdout__.write(str(sep).join(str(i) for i in args) + end)

    # Closes the bot, preventing all events.
    close = lambda self: setattr(self, "closed", True) or create_task(super().close())

    # A garbage collector for empty and unassigned objects in the database.
    @tracebacksuppressor(SemaphoreOverflowError)
    async def garbage_collect(self, obj):
        if not self.ready or hasattr(obj, "no_delete") or not any(hasattr(obj, i) for i in ("guild", "user", "channel", "garbage")) and not getattr(obj, "garbage_collect", None):
            return
        with MemoryTimer("gc_" + obj.name):
            async with obj._garbage_semaphore:
                data = obj.data
                if getattr(obj, "garbage_collect", None):
                    return await obj.garbage_collect()
                if len(data) <= 1024:
                    keys = data.keys()
                else:
                    low = xrand(ceil(len(data) / 1024)) << 10
                    keys = astype(data, alist).view[low:low + 1024]
                for key in keys:
                    if getattr(data, "unloaded", False):
                        return
                    if not key or type(key) is str:
                        continue
                    try:
                        # Database keys may be user, guild, or channel IDs
                        if getattr(obj, "channel", None):
                            d = self.get_channel(key)
                        elif getattr(obj, "user", None):
                            d = await self.fetch_user(key)
                        else:
                            if not data[key]:
                                raise LookupError
                            with suppress(KeyError):
                                d = self.cache.guilds[key]
                                continue
                            d = await self.fetch_messageable(key)
                        if d is not None:
                            continue
                    except:
                        print_exc()
                    print(f"Deleting {key} from {obj}...")
                    data.pop(key, None)

    # Calls a bot event, triggered by client events or others, across all bot databases. Calls may be sync or async.
    @tracebacksuppressor
    async def send_event(self, ev, *args, exc=False, **kwargs):
        if self.closed:
            return
        with MemoryTimer(ev):
            ctx = emptyctx if exc else tracebacksuppressor
            events = self.events.get(ev, ())
            if len(events) == 1:
                with ctx:
                    return await create_future(events[0](*args, **kwargs))
                return
            futs = [create_future(func(*args, **kwargs)) for func in events]
            out = deque()
            for fut in futs:
                with ctx:
                    res = await fut
                    out.append(res)
            return out

    # Gets the full list of invites from a guild, if applicable.
    @tracebacksuppressor(default=[])
    async def get_full_invites(self, guild):
        member = guild.get_member(self.id)
        if member.guild_permissions.create_instant_invite:
            invitedata = await Request(
                f"https://discord.com/api/{api}/guilds/{guild.id}/invites",
                authorise=True,
                aio=True,
                json=True,
            )
            invites = [cdict(invite) for invite in invitedata]
            return sorted(invites, key=lambda invite: (invite.max_age == 0, -abs(invite.max_uses - invite.uses), len(invite.url)))
        return []

    # Gets the first accessable text channel in the target guild.
    def get_first_sendable(self, guild, member):
        if member is None:
            return guild.owner
        found = {}
        for channel in sorted(guild.text_channels, key=lambda c: c.id):
            if channel.permissions_for(member).send_messages:
                with suppress(ValueError):
                    rname = full_prune(channel.name).replace("-", " ").replace("_", " ").split(maxsplit=1)[0]
                    i = ("miza", "bots", "bot", "general").index(rname)
                    if i < min(found):
                        found[i] = channel
        if found:
            return found[min(found)]
        channel = guild.system_channel
        if channel is None or not channel.permissions_for(member).send_messages:
            channel = guild.rules_channel
            if channel is None or not channel.permissions_for(member).send_messages:
                for channel in sorted(guild.text_channels, key=lambda c: c.id):
                    if channel.permissions_for(member).send_messages:
                        return channel
                return guild.owner
        return channel

    # Returns a discord object if it is in any of the internal cache.
    def in_cache(self, o_id):
        cache = self.cache
        try:
            return self.cache.users[o_id]
        except KeyError:
            pass
        try:
            return self.cache.channels[o_id]
        except KeyError:
            pass
        try:
            return self.cache.guilds[o_id]
        except KeyError:
            pass
        try:
            return self.cache.roles[o_id]
        except KeyError:
            pass
        try:
            return self.cache.emojis[o_id]
        except KeyError:
            pass
        try:
            return self.data.mimics[o_id]
        except KeyError:
            pass

    # Fetches either a user or channel object from ID, using the bot cache when possible.
    async def fetch_messageable(self, s_id):
        if type(s_id) is not int:
            try:
                s_id = int(s_id)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid messageable identifier: {s_id}")
        with suppress(KeyError):
            return self.get_user(s_id)
        with suppress(KeyError):
            return self.cache.channels[s_id]
        try:
            user = await super().fetch_user(s_id)
        except (LookupError, discord.NotFound):
            channel = await super().fetch_channel(s_id)
            self.cache.channels[s_id] = channel
            return channel
        self.cache.users[s_id] = user
        return user

    # Fetches a user from ID, using the bot cache when possible.
    async def _fetch_user(self, u_id):
        async with self.user_semaphore:
            user = await super().fetch_user(u_id)
            self.cache.users[u_id] = user
            return user
    def fetch_user(self, u_id):
        with suppress(KeyError):
            return as_fut(self.get_user(u_id))
        u_id = verify_id(u_id)
        if type(u_id) is not int:
            raise TypeError(f"Invalid user identifier: {u_id}")
        return self._fetch_user(u_id)

    async def auser2cache(self, u_id):
        with suppress(discord.NotFound):
            self.cache.users[u_id] = await super().fetch_user(u_id)

    def user2cache(self, data):
        users = self.cache.users
        u_id = int(data["id"])
        if u_id not in users:
            if type(data) is dict:
                with tracebacksuppressor:
                    if "s" in data:
                        s = data.pop("s")
                        data["username"], data["discriminator"] = s.rsplit("#", 1)
                    else:
                        s = data["username"] + "#" + data["discriminator"]
                    self.usernames[s] = users[u_id] = self._state.store_user(data)
                    return
            self.user_loader.add(u_id)

    def update_users(self):
        if self.user_loader:
            if not self.user_semaphore.busy:
                u_id = self.user_loader.pop()
                if u_id not in self.cache.users:
                    create_task(self.auser2cache(u_id))

    # Gets a user from ID, using the bot cache.
    def get_user(self, u_id, replace=False):
        if type(u_id) is not int:
            try:
                u_id = int(u_id)
            except (ValueError, TypeError):
                user = self.user_from_identifier(u_id)
                if user is not None:
                    return user
                if "#" in u_id:
                    raise LookupError(f"User identifier not found: {u_id}")
                u_id = verify_id(u_id)
                if not isinstance(u_id, int):
                    raise TypeError(f"Invalid user identifier: {u_id}")
        with suppress(KeyError):
            return self.cache.users[u_id]
        if u_id == self.deleted_user:
            user = self.GhostUser()
            user.system = True
            user.name = "Deleted User"
            user.nick = "Deleted User"
            user.id = u_id
        else:
            try:
                user = super().get_user(u_id)
                if user is None:
                    raise LookupError
            except:
                if replace:
                    return self.get_user(self.deleted_user)
                raise KeyError("Target user ID not found.")
        self.cache.users[u_id] = user
        return user

    async def find_users(self, argl, args, user, guild, roles=False):
        if not argl and not args:
            return (user,)
        if argl:
            users = {}
            for u_id in argl:
                u = await self.fetch_user_member(u_id, guild)
                users[u.id] = u
            return users.values()
        u_id = verify_id(args.pop(0))
        if type(u_id) is int:
            role = guild.get_role(u_id)
            if role is not None:
                if roles:
                    return (role,)
                return role.members
        if type(u_id) is str and "@" in u_id and ("everyone" in u_id or "here" in u_id):
            return await self.get_full_members(guild)
        u = await self.fetch_user_member(u_id, guild)
        return (u,)

    def user_from_identifier(self, u_id):
        if "#" in u_id:
            spl = u_id.split()
            for i in range(len(spl)):
                uid = " ".join(spl[i:])
                try:
                    return self.usernames[uid]
                except:
                    pass

    async def fetch_user_member(self, u_id, guild=None):
        u_id = verify_id(u_id)
        if type(u_id) is int:
            try:
                user = self.cache.users[u_id]
            except KeyError:
                try:
                    user = await self.fetch_user(u_id)
                except discord.NotFound:
                    if "webhooks" in self.data:
                        for channel in guild.text_channels:
                            webhooks = await self.data.webhooks.get(channel)
                            try:
                                return [w for w in webhooks if w.id == u_id][0]
                            except IndexError:
                                pass
                    raise
            with suppress():
                if guild:
                    member = guild.get_member(user.id)
                    if member is not None:
                        return member
            with suppress():
                return self.get_member(u_id, guild, find_others=False)
            return user
        user = self.user_from_identifier(u_id)
        if user is not None:
            if guild is None:
                return user
            member = guild.get_member(user.id)
            if member is not None:
                return member
            return user
        return await self.fetch_member_ex(u_id, guild)

    async def get_full_members(self, guild):
        members = guild._members.values()
        if "bans" in self.data:
            members = set(members)
            for b in self.data.bans.get(guild.id, ()):
                try:
                    user = await self.fetch_user(b.get("u", self.deleted_user))
                except LookupError:
                    user = self.cache.users[self.deleted_user]
                members.add(user)
        return members

    async def query_members(self, members, query, fuzzy=0.5):
        query = str(query)
        fuz_base = None if fuzzy is None else 0
        with suppress(LookupError):
            return await str_lookup(
                members,
                query,
                qkey=userQuery1,
                ikey=userIter1,
                loose=False,
                fuzzy=fuz_base,
            )
        with suppress(LookupError):
            return await str_lookup(
                members,
                query,
                qkey=userQuery2,
                ikey=userIter2,
                fuzzy=fuz_base,
            )
        with suppress(LookupError):
            return await str_lookup(
                members,
                query,
                qkey=userQuery3,
                ikey=userIter3,
                fuzzy=fuz_base,
            )
        with suppress(LookupError):
            return await str_lookup(
                members,
                query,
                qkey=userQuery4,
                ikey=userIter4,
                fuzzy=fuzzy,
            )
        raise LookupError(f"No results for {query}.")

    # Fetches a member in the target server by ID or name lookup.
    async def fetch_member_ex(self, u_id, guild=None, allow_banned=True, fuzzy=1 / 3):
        if type(u_id) is not int and u_id.isnumeric():
            with suppress(TypeError, ValueError):
                u_id = int(u_id)
        member = None
        if type(u_id) is int:
            member = guild.get_member(u_id)
        if member is None:
            if type(u_id) is int:
                with suppress(LookupError):
                    if guild:
                        member = await self.fetch_member(u_id, guild)
                    else:
                        member = await self.fetch_user(u_id)
            if member is None:
                if allow_banned:
                    members = await self.get_full_members(guild)
                else:
                    members = guild.members
                if not members:
                    members = guild.members = await guild.fetch_members(limit=None)
                    guild._members.update({m.id: m for m in members})
                return await self.query_members(members, u_id, fuzzy=fuzzy)
        return member

    # Fetches the first seen instance of the target user as a member in any shared server.
    def fetch_member(self, u_id, guild=None, find_others=False):
        return create_future(self.get_member, u_id, guild, find_others)

    def get_member(self, u_id, guild=None, find_others=True):
        if type(u_id) is not int:
            try:
                u_id = int(u_id)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid user identifier: {u_id}")
        if find_others:
            with suppress(LookupError):
                member = self.cache.members[u_id].guild.get_member(u_id)
                if member is None:
                    raise LookupError
                return member
        g = self.cache.guilds
        if guild is None:
            if find_others:
                guilds = deque(self.cache.guilds.values())
            else:
                return self.cache.users[u_id]
        else:
            if find_others:
                guilds = deque(g[i] for i in g if g[i].id != guild.id)
                guilds.appendleft(guild)
            else:
                guilds = [guild]
        member = None
        for guild in guilds:
            member = guild.get_member(u_id)
            if member is not None:
                break
        if member is None:
            raise LookupError("Unable to find member data.")
        if find_others:
            self.cache.members[u_id] = member
        return member

    # Fetches a guild from ID, using the bot cache when possible.
    async def fetch_guild(self, g_id, follow_invites=True):
        if type(g_id) is not int:
            try:
                g_id = int(g_id)
            except (ValueError, TypeError):
                if follow_invites:
                    try:
                        # Parse and follow invites to get partial guild info
                        invite = await super().fetch_invite(g_id.strip("< >"))
                        g = invite.guild
                        with suppress(KeyError):
                            return self.cache.guilds[g.id]
                        if not hasattr(g, "member_count"):
                            guild = cdict(ghost=True, member_count=invite.approximate_member_count)
                            for at in g.__slots__:
                                setattr(guild, at, getattr(g, at))
                            icon = str(guild.icon)
                            guild.icon_url = f"https://cdn.discordapp.com/icons/{guild.id}/{icon}"
                            if icon.startswith("a_"):
                                guild.icon_url += ".gif"
                            guild.created_at = snowflake_time_3(guild.id)
                        else:
                            guild = g
                        return guild
                    except (discord.NotFound, discord.HTTPException) as ex:
                        raise LookupError(str(ex))
                raise TypeError(f"Invalid server identifier: {g_id}")
        with suppress(KeyError):
            return self.cache.guilds[g_id]
        try:
            guild = super().get_guild(g_id)
            if guild is None:
                raise EOFError
        except:
            guild = await super().fetch_guild(g_id)
        # self.cache.guilds[g_id] = guild
        return guild

    # Fetches a channel from ID, using the bot cache when possible.
    async def _fetch_channel(self, c_id):
        channel = await super().fetch_channel(c_id)
        self.cache.channels[c_id] = channel
        return channel
    def fetch_channel(self, c_id):
        if type(c_id) is not int:
            try:
                c_id = int(c_id)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid channel identifier: {c_id}")
        with suppress(KeyError):
            return as_fut(self.cache.channels[c_id])
        return self._fetch_channel(c_id)

    def force_channel(self, data):
        if isinstance(data, dict):
            c_id = data["channel_id"]
        else:
            c_id = verify_id(data)
        if type(c_id) is not int:
            try:
                c_id = int(c_id)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid channel identifier: {c_id}")
        with suppress(KeyError):
            return self.cache.channels[c_id]
        channel, _ = bot._get_guild_channel(dict(channel_id=c_id))
        if channel and type(channel) is not discord.Object:
            if not isinstance(channel, discord.abc.PrivateChannel):
                self.cache.channels[c_id] = channel
            return channel
        channel = self.cache.channels[c_id] = cdict(
            id=c_id,
            name=f"<#{c_id}>",
            mention=f"<#{c_id}>",
            guild=None,
            _data=dict(id=c_id),
            _state=self._state,
            _members={self.id: self.user},
            _type=11,
            type=11,
            guild_id=None,
            owner_id=self.id,
            owner=self.user,
            parent_id=None,
            parent=None,
            last_message_id=None,
            message_count=1,
            member_count=1,
            slowmode_delay=0,
            me=self.user,
            locked=False,
            archived=False,
            archiver_id=None,
            auto_archive_duration=inf,
            archive_timestamp=inf,
            permissions_for=lambda *args: 0,
            join=async_nop,
            leave=async_nop,
        )
        channel.update(dict(
            thread=channel,
            _get_channel=lambda: as_fut(channel),
            is_private=lambda: channel._type == 12,
            is_news=lambda: channel._type == 10,
            is_nsfw=lambda: is_nsfw(channel.parent),
            delete_messages=lambda *args, **kwargs: discord.channel.TextChannel.delete_messages(channel, *args, **kwargs),
            purge=lambda *args, **kwargs: discord.channel.TextChannel.purge(channel, *args, **kwargs),
            edit=lambda *args, **kwargs: discord.channel.TextChannel.edit(channel, *args, **kwargs),
            add_user=lambda user: Request(
                f"https://discord.com/api/{api}/channels/{channel.id}/thread-members/{verify_id(user)}",
                method="PUT",
                authorise=True,
                aio=True,
            ),
            remove_user=lambda user: Request(
                f"https://discord.com/api/{api}/channels/{channel.id}/thread-members/{verify_id(user)}",
                method="DELETE",
                authorise=True,
                aio=True,
            ),
            delete=lambda reason=None: discord.abc.GuildChannel.delete(channel, reason=reason),
            _add_member=lambda member: self._members.__setitem__(member.id, member),
            _pop_member=lambda m_id: self._members.pop(m_id, None),
            send=lambda *args, **kwargs: discord.abc.Messageable.send(channel, *args, **kwargs),
            trigger_typing=lambda: discord.abc.Messageable.trigger_typing(channel),
            typing=lambda: discord.abc.Messageable.typing(channel),
            fetch_message=lambda id: discord.abc.Messageable.fetch_message(channel, id),
            pins=lambda: discord.abc.Messageable.pins(channel),
            history=lambda *args, **kwargs: discord.abc.Messageable.history(channel, *args, **kwargs),
        ))
        create_task(self.manage_thread(channel))
        return channel

    async def manage_thread(self, channel):
        Request(
            f"https://discord.com/api/{api}/channels/{channel.id}/thread-members/@me",
            method="POST",
            authorise=True,
            aio=True,
        )
        data = await Request(
            f"https://discord.com/api/{api}/channels/{channel.id}",
            authorise=True,
            aio=True,
            json=True,
        )
        channel.guild_id = int(data["guild_id"])
        channel.guild = self.get_guild(channel.guild_id)
        channel.parent_id = int(data.get("parent_id", 0))
        channel.parent = channel.guild.get_channel(channel.parent_id) or self.get_channel(channel.parent_id)
        channel.permissions_for = channel.parent.permissions_for
        channel.owner_id = int(data.get("owner_id", 0))
        channel.owner = channel.guild.get_member(channel.owner_id) or self.get_user(channel.owner_id)
        channel.type = data["type"]
        channel.name = data["name"]
        channel.last_message_id = int(data.get("last_message_id") or 0) or None
        channel.slowmode_delay = data.get("rate_limit_per_user") or 0
        channel.message_count = data.get("message_count", 0)
        channel.member_count = data.get("member_count", 0)
        meta = data.get("thread_metadata", {})
        channel.update(meta)
        if channel.get("archiver_id"):
            channel.archiver_id = int(channel.archiver_id)
        channel.locked = channel.get("locked")

    # Fetches a message from ID and channel, using the bot cache when possible.
    async def _fetch_message(self, m_id, channel=None):
        if channel is None:
            raise LookupError("Message data not found.")
        with suppress(TypeError):
            int(channel)
            channel = await self.fetch_channel(channel)
        history = discord.abc.Messageable.history(channel, limit=101, around=cdict(id=m_id))
        messages = await history.flatten()
        data = {m.id: m for m in messages}
        self.cache.messages.update(data)
        return self.cache.messages[m_id]
    def fetch_message(self, m_id, channel=None):
        if type(m_id) is not int:
            try:
                m_id = int(m_id)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid message identifier: {m_id}")
        with suppress(KeyError):
            return as_fut(self.cache.messages[m_id])
        if "message_cache" in self.data:
            with suppress(KeyError):
                return as_fut(self.data.message_cache.load_message(m_id))
        return self._fetch_message(m_id, channel)

    # Fetches a role from ID and guild, using the bot cache when possible.
    async def fetch_role(self, r_id, guild=None):
        if type(r_id) is not int:
            try:
                r_id = int(r_id)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid role identifier: {r_id}")
        with suppress(KeyError):
            return self.cache.roles[r_id]
        try:
            role = guild.get_role(r_id)
            if role is None:
                raise EOFError
        except:
            if len(guild.roles) <= 1:
                roles = await guild.fetch_roles()
                guild.roles = sorted(roles)
                role = utils.get(roles, id=r_id)
            if role is None:
                raise LookupError("Role not found.")
        self.cache.roles[r_id] = role
        return role

    # Fetches an emoji from ID and guild, using the bot cache when possible.
    async def fetch_emoji(self, e_id, guild=None):
        if type(e_id) is not int:
            try:
                e_id = int(e_id)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid emoji identifier: {e_id}")
        with suppress(KeyError):
            return self.cache.emojis[e_id]
        try:
            emoji = super().get_emoji(e_id)
            if emoji is None:
                raise EOFError
        except:
            if guild is not None:
                emoji = await guild.fetch_emoji(e_id)
            else:
                raise LookupError("Emoji not found.")
        self.cache.emojis[e_id] = emoji
        return emoji

    # Searches the bot database for a webhook mimic from ID.
    def get_mimic(self, m_id, user=None):
        if "mimics" in self.data:
            with suppress(KeyError):
                with suppress(ValueError, TypeError):
                    m_id = "&" + str(int(m_id))
                mimic = self.data.mimics[m_id]
                return mimic
            if user is not None:
                with suppress(KeyError):
                    mimics = self.data.mimics[user.id]
                    mlist = mimics[m_id]
                    return self.get_mimic(choice(mlist))
        raise LookupError("Unable to find target mimic.")

    # Gets the DM channel for the target user, creating a new one if none exists.
    async def get_dm(self, user):
        if isinstance(user, discord.abc.PrivateChannel):
            return user
        with suppress(TypeError):
            int(user)
            user = await self.fetch_user(user)
        channel = user.dm_channel
        if channel is None:
            channel = await user.create_dm()
        return channel

    def get_available_guild(self, animated=True):
        found = [{} for _ in loop(5)]
        for guild in self.guilds:
            m = guild.me
            if m is not None and m.guild_permissions.manage_emojis:
                owners_in = self.owners.intersection(guild._members)
                if guild.owner_id == self.id:
                    x = 0
                elif len(owners_in) == len(self.owners):
                    x = 1
                elif owners_in:
                    x = 2
                elif m.guild_permissions.administrator and len(deque(member for member in guild.members if not member.bot)) <= 5:
                    x = 3
                else:
                    x = 4
                if animated:
                    rem = guild.emoji_limit - len(deque(e for e in guild.emojis if e.animated))
                    rem /= len(guild.members)
                    if rem > 0:
                        found[x][rem] = guild
                else:
                    rem = guild.emoji_limit - len(deque(e for e in guild.emojis if not e.animated))
                    rem /= len(guild.members)
                    if rem > 0:
                        found[x][rem] = guild
        for i, f in enumerate(found):
            if f:
                return f[max(f.keys())]
        raise LookupError("Unable to find suitable guild.")

    @tracebacksuppressor
    def create_progress_bar(self, length, ratio=0.5):
        if "emojis" in self.data:
            return create_future(self.data.emojis.create_progress_bar, length, ratio)
        position = min(length, round(length * ratio))
        return as_fut("⬜" * position + "⬛" * (length - position))

    async def history(self, channel, limit=200, before=None, after=None, care=True):
        c = self.in_cache(verify_id(channel))
        if c is None:
            c = channel
        if channel is None:
            return
        if not is_channel(channel):
            channel = await self.get_dm(channel)
        found = set()
        if "channel_cache" in self.data:
            async for message in self.data.channel_cache.get(channel.id, as_message=care, force=False):
                if isinstance(message, int):
                    message = cdict(id=message)
                if before:
                    if message.id >= time_snowflake(before):
                        continue
                if after:
                    if message.id <= time_snowflake(after):
                        break
                found.add(message.id)
                yield message
                if limit is not None and len(found) >= limit:
                    return
        if type(before) is int:
            before = cdict(id=before)
        if type(after) is int:
            after = cdict(id=after)
        if getattr(channel, "simulated", None):
            return
        async for message in discord.abc.Messageable.history(channel, limit=limit, before=before, after=after):
            if message.id not in found:
                self.add_message(message, files=False, force=True)
                yield message

    async def get_last_message(self, channel, key=None):
        m_id = getattr(channel, "last_message_id", None)
        if m_id:
            try:
                return await self.fetch_message(m_id, channel)
            except:
                pass
        if key:
            async for message in self.history(channel):
                if key(message):
                    return message
        async for message in self.history(channel):
            return message

    async def get_last_image(self, channel):
        async for message in self.history(channel):
            try:
                return get_last_image(message)
            except FileNotFoundError:
                pass
        raise FileNotFoundError("Image file not found.")

    mime = magic.Magic(mime=True, mime_encoding=True)
    mimes = {}

    async def id_from_message(self, m_id):
        if not m_id:
            return m_id
        m_id = as_str(m_id)
        links = await self.follow_url(m_id)
        m_id = links[0] if links else m_id
        if is_url(m_id.strip("<>")) and "/emojis/" in m_id:
            return int(m_id.strip("<>").split("/emojis/", 1)[-1].split(".", 1)[0])
        if m_id[0] == "<" and m_id[-1] == ">" and ":" in m_id:
            n = m_id.rsplit(":", 1)[-1][:-1]
            if n.isnumeric():
                return int(n)
        if m_id.isnumeric():
            m_id = int(m_id)
            if m_id in self.cache.messages:
                return await self.id_from_message(self.cache.messages[m_id].content)
        return verify_id(m_id)

    # Finds URLs in a string, following any discord message links found.
    async def follow_url(self, url, it=None, best=False, preserve=True, images=True, allow=False, limit=None):
        if limit is not None and limit <= 0:
            return []
        if it is None:
            urls = find_urls(url)
            if not urls:
                return []
            it = {}
        else:
            urls = [url]
        out = deque()
        if preserve or allow:
            lost = deque()
        else:
            lost = None
        if images:
            medias = ("video", "image", "thumbnail")
        else:
            medias = "video"
        for url in urls:
            u = getattr(url, "url", None)
            if u:
                url = u
            if is_discord_message_link(url):
                found = deque()
                spl = url[url.index("channels/") + 9:].replace("?", "/").split("/", 2)
                c = await self.fetch_channel(spl[1])
                m = await self.fetch_message(spl[2], c)
                # All attachments should be valid URLs
                if best:
                    found.extend(best_url(a) for a in m.attachments)
                else:
                    found.extend(a.url for a in m.attachments)
                found.extend(find_urls(m.content))
                temp = await self.follow_to_image(m.content)
                found.extend(filter(is_url, temp))
                # Attempt to find URLs in embed contents
                for e in m.embeds:
                    for a in medias:
                        obj = getattr(e, a, None)
                        if obj:
                            if best:
                                url = best_url(obj)
                            else:
                                url = obj.url
                            if url:
                                found.append(url)
                                break
                # Attempt to find URLs in embed descriptions
                [found.extend(find_urls(e.description)) for e in m.embeds if e.description]
                if images:
                    for r in m.reactions:
                        e = r.emoji
                        if hasattr(e, "url"):
                            found.append(as_str(e.url))
                        else:
                            u = translate_emojis(e)
                            if is_url(u):
                                found.append(u)
                    if not found:
                        m = await c.fetch_message(int(spl[2]))
                        self.bot.add_message(m, files=False, force=True)
                        for r in m.reactions:
                            e = r.emoji
                            if hasattr(e, "url"):
                                found.append(as_str(e.url))
                            else:
                                u = translate_emojis(e)
                                if is_url(u):
                                    found.append(u)
                for u in found:
                    # Do not attempt to find the same URL twice
                    if u not in it:
                        it[u] = True
                        if not len(it) & 255:
                            await asyncio.sleep(0.2)
                        found2 = await self.follow_url(u, it, best=best, preserve=preserve, images=images, allow=allow, limit=limit)
                        if len(found2):
                            out.extend(found2)
                        elif allow and m.content:
                            lost.append(m.content)
                        elif preserve:
                            lost.append(u)
            elif images and is_imgur_url(url):
                first = url.split("?", 1)[0]
                if not first.endswith(".jpg"):
                    first += ".jpg"
                out.append(first)
            elif images and is_giphy_url(url):
                first = url.split("?", 1)[0]
                item = first[first.rindex("/") + 1:]
                out.append(f"https://media2.giphy.com/media/{item}/giphy.gif")
            elif images and is_reddit_url(url):
                first = url.split("?", 1)[0]
                b = await Request(url, aio=True, timeout=16)
                search = b'<script id="data">window.___r = '
                with tracebacksuppressor:
                    b = b[b.index(search) + len(search):]
                    b = b[:b.index(b";</script><")]
                    data = orjson.loads(b)
                    for model in data["posts"]["models"].values():
                        try:
                            stream = model["media"]["scrubberThumbSource"]
                        except KeyError:
                            continue
                        else:
                            found = True
                            out.append(stream)
                            break
            elif images and not any(maps((is_discord_url, is_youtube_url, is_youtube_stream), url)):
                found = False
                if images or is_tenor_url(url) or is_deviantart_url(url) or self.is_webserver_url(url):
                    skip = False
                    if url in self.mimes:
                        skip = "text/html" not in self.mimes[url]
                    if not skip:
                        resp = await create_future(reqs.next().get, url, headers=Request.header(), stream=True)
                        self.activity += 1
                        url = as_str(resp.url)
                        head = fcdict(resp.headers)
                        ctype = [t.strip() for t in head.get("Content-Type", "").split(";")]
                        if "text/html" in ctype:
                            it = resp.iter_content(65536)
                            data = await create_future(next, it)
                            s = as_str(data)
                            try:
                                s = s[s.index("<meta") + 5:]
                                search = 'http-equiv="refresh" content="'
                                try:
                                    s = s[s.index(search) + len(search):]
                                    s = s[:s.index('"')]
                                    res = None
                                    for k in s.split(";"):
                                        temp = k.strip()
                                        if temp.casefold().startswith("url="):
                                            res = temp[4:]
                                            break
                                    if not res:
                                        raise ValueError
                                except ValueError:
                                    search ='property="og:image" content="'
                                    s = s[s.index(search) + len(search):]
                                    res = s[:s.index('"')]
                            except ValueError:
                                pass
                            else:
                                found = True
                                print(res)
                                out.append(res)
                if not found:
                    out.append(url)
        if lost:
            out.extend(lost)
        if not out:
            return urls
        if limit is not None:
            return list(out)[:limit]
        return out

    def detect_mime(self, url):
        try:
            return self.mimes[url]
        except KeyError:
            pass
        resp = reqs.next().get(url, stream=True)
        self.activity += 1
        head = fcdict(resp.headers)
        try:
            mime = [t.strip() for t in head.get("Content-Type", "").split(";")]
        except KeyError:
            it = resp.iter_content(65536)
            data = next(it)
            mime = [t.strip() for t in self.mime.from_buffer(data).split(";")]
        self.mimes[url] = mime
        return mime

    emoji_stuff = {}

    def is_animated(self, e, verify=False):
        if type(e) in (int, str):
            try:
                emoji = self.cache.emojis[e]
            except KeyError:
                e = int(e)
                if e <= 0 or e > time_snowflake(dtn(), high=True):
                    return
                try:
                    return self.emoji_stuff[e]
                except KeyError:
                    pass
                base = f"https://cdn.discordapp.com/emojis/{e}."
                if verify:
                    fut = create_future_ex(Request, base + "png")
                url = base + "gif"
                with reqs.next().head(url, stream=True) as resp:
                    self.activity += 1
                    if resp.status_code in range(400, 500):
                        if not verify:
                            return False
                        try:
                            fut.result()
                        except ConnectionError:
                            self.emoji_stuff[e] = None
                            return
                        self.emoji_stuff[e] = False
                        return False
                self.emoji_stuff[e] = True
                return True
        else:
            emoji = e
        return emoji.animated
    
    def emoji_exists(self, e):
        if type(e) in (int, str):
            url = f"https://cdn.discordapp.com/emojis/{e}.png"
            with reqs.next().head(url, stream=True) as resp:
                self.activity += 1
                if resp.status_code in range(400, 500):
                    self.emoji_stuff.pop(int(e), None)
                    return
        else:
            if e.id not in self.cache.emojis:
                return
        return True

    async def min_emoji(self, e):
        animated = await create_future(self.is_animated, e, verify=True)
        if animated is None:
            raise LookupError(f"Emoji {e} does not exist.")
        if type(e) in (int, str):
            e = cdict(id=e, animated=animated)
        return min_emoji(e)

    # Follows a message link, replacing emojis and user mentions with their icon URLs.
    async def follow_to_image(self, url):
        temp = find_urls(url)
        if temp:
            return temp
        users = find_users(url)
        emojis = find_emojis(url)
        out = deque()
        if users:
            futs = [create_task(self.fetch_user(verify_id(u))) for u in users]
            for fut in futs:
                with suppress(LookupError):
                    res = await fut
                    out.append(best_url(res))
        for s in emojis:
            s = s[3:]
            i = s.index(":")
            e_id = int(s[i + 1:s.rindex(">")])
            try:
                out.append(str(self.cache.emojis[e_id].url))
            except KeyError:
                animated = await create_future(self.is_animated, e_id)
                if animated:
                    end = "gif"
                else:
                    end = "png"
                url = f"https://cdn.discordapp.com/emojis/{e_id}.{end}"
            out.append(url)
        if not out:
            out = find_urls(translate_emojis(replace_emojis(url)))
        return out

        # Sends a message to a channel, then edits to add links to all attached files.
    async def send_with_file(self, channel, msg=None, file=None, filename=None, embed=None, best=False, rename=True, reference=None, reacts=""):
        if not msg:
            msg = ""
        f = None
        fsize = 0
        size = 8388608
        with suppress(AttributeError):
            size = channel.guild.filesize_limit
        if getattr(channel, "simulated", None) or getattr(channel, "guild", None) and not channel.permissions_for(channel.guild.me).attach_files:
            size = -1
        data = file
        if file and not hasattr(file, "fp"):
            if type(file) is str:
                if not os.path.exists(file):
                    raise FileNotFoundError(file)
                fsize = os.path.getsize(file)
                f = file
            else:
                data  = file
                file = CompatFile(data, filename)
                fsize = len(data)
        if fsize <= size:
            if not hasattr(file, "fp"):
                f2 = CompatFile(file, filename)
            else:
                f2 = file
            if not filename:
                filename = file
            file = f2
            fp = file.fp
            fp.seek(0)
            data = fp.read()
            fsize = len(data)
            fp.seek(0)
            with suppress(AttributeError):
                fp.clear()
        # if reference and getattr(reference, "slash", None) or getattr(reference, "simulated", None):
        #     reference = None
        try:
            if fsize > size:
                if not f:
                    f = filename if filename and not hasattr(file, "fp") else getattr(file, "_fp", None) or data
                if type(f) is not str:
                    f = as_str(f)
                if "." in f:
                    ext = f.rsplit(".", 1)[-1]
                else:
                    ext = None
                urls = await create_future(as_file, file if getattr(file, "_fp", None) else f, filename=filename, ext=ext, rename=rename)
                if hasattr(channel, "simulated"):
                    urls = (urls[1],)
                message = await send_with_reply(channel, reference, (msg + ("" if msg.endswith("```") else "\n") + urls[0]).strip(), embed=embed)
            else:
                message = await send_with_reply(channel, reference, msg, embed=embed, file=file)
                if filename is not None:
                    if hasattr(filename, "filename"):
                        filename = filename.filename
                    with suppress():
                        os.remove(filename)
        except:
            if filename is not None:
                if type(filename) is not str:
                    filename = getattr(filename, "filename", None) or filename.name
                print(filename, os.path.getsize(filename))
                with suppress():
                    os.remove(filename)
            raise
        if not getattr(reference, "slash", None) and message.attachments:
            await self.add_attachment(message.attachments[0], data)
            content = message.content + ("" if message.content.endswith("```") else "\n") + ("\n".join("<" + best_url(a) + ">" for a in message.attachments) if best else "\n".join("<" + a.url + ">" for a in message.attachments))
            message = await message.edit(content=content.strip())
        if not message:
            print("No message detected.")
        elif reacts:
            for react in reacts:
                await message.add_reaction(react)
        return message

    # Inserts a message into the bot cache, discarding existing ones if full.
    def add_message(self, message, files=True, cache=True, force=False):
        if self.closed:
            return message
        try:
            m = self.cache.messages[message.id]
        except KeyError:
            m = None
        else:
            if force < 2 and isinstance(m, self.ExtendedMessage):
                with suppress(AttributeError):
                    if not object.__getattribute__(m, "replaceable"):
                        return m
            if getattr(m, "slash", False):
                message.slash = m.slash
        if cache and not m or force:
            created_at = message.created_at
            if created_at.tzinfo:
                created_at = created_at.replace(tzinfo=None)
            if not getattr(message, "simulated", None) and "channel_cache" in self.data:
                self.data.channel_cache.add(message.channel.id, message.id)
                if message.guild and hasattr(message.author, "guild"):
                    guild = message.guild
                    author = message.author
                    try:
                        if guild._members[author.id] != author:
                            guild._members[author.id] = author
                            if "guilds" in self.data:
                                self.data.guilds.register(guild, force=False)
                    except KeyError:
                        pass
            if files and not message.author.bot:
                if (utc_dt() - created_at).total_seconds() < 7200:
                    for attachment in message.attachments:
                        create_task(self.add_and_test(message, attachment))
                    for url in find_urls(message.content):
                        if is_discord_url(url) and "attachments/" in url:
                            attachment = cdict(id=url.rsplit("/", 2)[-2], url=url, read=lambda: self.get_request(url))
                            create_task(self.add_and_test(message, attachment))
            self.cache.messages[message.id] = message
            if (utc_dt() - created_at).total_seconds() < 86400 * 14 and "message_cache" in self.data and not getattr(message, "simulated", None):
                self.data.message_cache.save_message(message)
        return message

    # Deletes a message from the bot cache.
    def remove_message(self, message):
        self.cache.messages.pop(message.id, None)
        if not message.author.bot:
            s = message_repr(message, username=True)
            ch = f"deleted/{message.channel.id}.txt"
            print(s, file=ch)

    async def add_attachment(self, attachment, data=None):
        if attachment.id not in self.cache.attachments:
            self.cache.attachments[attachment.id] = None
            if data is None:
                data = await attachment.read()
            self.cache.attachments[attachment.id] = data
            fn = f"cache/attachment_{attachment.id}.bin"
            if not os.path.exists(fn):
                with open(fn, "wb") as f:
                    await create_future(f.write, data)
        return attachment

    async def add_and_test(self, message, attachment):
        attachment = await self.add_attachment(attachment)
        if "prot" in self.data:
            fn = f"cache/attachment_{attachment.id}.bin"
            if fn in self.cache.attachments:
                if self.cache.attachments[fn]:
                    await self.data.prot.call(message, fn, known=self.cache.attachments[fn])
                return
            if not os.path.exists(fn):
                data = await self.get_attachment(str(attachment.url))
                if not data:
                    return
                with open(fn, "wb") as f:
                    await create_future(f.write, data)
            if get_mime(fn).startswith("image/"):
                res = await self.data.prot.call(message, fn)
            else:
                res = ""
            self.cache.attachments[fn] = res

    def attachment_from_file(self, file):
        a_id = int(file.split(".", 1)[0][11:])
        self.cache.attachments[a_id] = a_id

    async def get_attachment(self, url, full=True, allow_proxy=False):
        if not is_discord_url(url) or "attachments/" not in url[:64]:
            return
        with suppress(ValueError):
            a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
            with suppress(LookupError):
                for i in range(30):
                    data = self.cache.attachments[a_id]
                    if data is not None:
                        if isinstance(data, memoryview):
                            self.cache.attachments[a_id] = data = bytes(data)
                        if not isinstance(data, bytes):
                            self.cache.attachments[a_id] = None
                            try:
                                with open(f"cache/attachment_{data}.bin", "rb") as f:
                                    data = await create_future(f.read)
                            except FileNotFoundError:
                                if allow_proxy and is_image(url):
                                    url = to_png(url)
                                data = await Request(url, aio=True)
                                await self.add_attachment(cdict(id=a_id), data=data)
                                return data
                            else:
                                self.cache.attachments[a_id] = data
                        print(f"Successfully loaded attachment {a_id} from cache.")
                        return data
                    if i:
                        await asyncio.sleep(0.25 * i)
            if allow_proxy and is_image(url):
                url = to_png(url)
            if full:
                data = await Request(url, aio=True)
                await self.add_attachment(cdict(id=a_id), data=data)
                return data
            return await create_future(reqs.next().get, url, stream=True)
        return

    async def get_request(self, url, limit=None, full=True, timeout=12):
        fn = is_file(url)
        if fn:
            if limit:
                size = os.path.getsize(fn)
                if size > limit:
                    raise OverflowError(f"Supplied file too large ({size}) > ({limit})")
            if not full:
                return open(url, "rb")
            with open(url, "rb") as f:
                return await create_future(f.read)
        data = await self.get_attachment(url, full=full)
        if data is not None:
            return data
        if not full:
            return await create_future(reqs.next().get, url, stream=True)
        return await Request(url, timeout=timeout, aio=True)

    def get_colour(self, user):
        if user is None:
            return as_fut(16777214)
        if hasattr(user, "icon_url"):
            user = astype(user.icon_url, str)
        url = worst_url(user)
        return self.data.colours.get(url)

    async def get_proxy_url(self, user, force=False):
        if hasattr(user, "webhook"):
            url = user.webhook.avatar_url_as(format="webp", size=4096)
        else:
            with tracebacksuppressor:
                url = best_url(user)
        if not url:
            return self.discord_icon
        if "proxies" in self.data:
            with tracebacksuppressor:
                url = (await self.data.exec.uproxy(url, force=force)) or url
        return url

    async def as_embed(self, message, link=False, colour=False):
        emb = discord.Embed(description="").set_author(**get_author(message.author))
        if colour:
            col = await self.get_colour(message.author)
            emb.colour = col
        content = message.content or message.system_content
        if not content:
            if len(message.attachments) == 1:
                url = message.attachments[0].url
                if is_image(url):
                    url = await self.data.exec.uproxy(url)
                    emb.url = url
                    emb.set_image(url=url)
                    if link:
                        emb.description = lim_str(f"{emb.description}\n\n[View Message](https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id})", 4096)
                        emb.timestamp = message.edited_at or message.created_at
                    return emb
            elif not message.attachments and len(message.embeds) == 1:
                emb2 = message.embeds[0]
                if emb2.description != EmptyEmbed and emb2.description:
                    emb.description = emb2.description
                if emb2.title:
                    emb.title = emb2.title
                if emb2.url:
                    emb.url = emb2.url
                if emb2.image:
                    url = await self.data.exec.uproxy(emb2.image.url)
                    emb.set_image(url=url)
                if emb2.thumbnail:
                    url = await self.data.exec.uproxy(emb2.thumbnail.url)
                    emb.set_thumbnail(url=url)
                for f in emb2.fields:
                    if f:
                        emb.add_field(name=f.name, value=f.value, inline=getattr(f, "inline", True))
                if link:
                    emb.description = lim_str(f"{emb.description}\n\n[View Message](https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id})", 4096)
                    emb.timestamp = message.edited_at or message.created_at
                return emb
        else:
            urls = await self.follow_url(content)
            if urls:
                with tracebacksuppressor:
                    url = urls[0]
                    resp = await create_future(reqs.next().get, url, headers=Request.header(), _timeout_=12)
                    self.activity += 1
                    headers = fcdict(resp.headers)
                    if headers.get("Content-Type", "").split("/", 1)[0] == "image":
                        if float(headers.get("Content-Length", inf)) < 8388608:
                            url = await self.data.exec.uproxy(url)
                        else:
                            url = await self.data.exec.aproxy(url)
                        emb.url = url
                        emb.set_image(url=url)
                        if url != content:
                            emb.description = content
                        if link:
                            emb.description = lim_str(f"{emb.description}\n\n[View Message](https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id})", 4096)
                            emb.timestamp = message.edited_at or message.created_at
                        return emb
        emb.description = content
        if len(message.embeds) > 1 or content:
            urls = list(chain(("(" + e.url + ")" for e in message.embeds[1:] if e.url), ("[" + best_url(a) + "]" for a in message.attachments)))
            items = []
            for i in range((len(urls) + 9) // 10):
                temp = urls[i * 10:i * 10 + 10]
                temp2 = await self.data.exec.uproxy(*temp, collapse=False)
                items.extend(temp2[x] or temp[x] for x in range(len(temp)))
        else:
            items = None
        if items:
            if emb.description in items:
                emb.description = lim_str("\n".join(items), 4096)
            elif emb.description or items:
                emb.description = lim_str(emb.description + "\n" + "\n".join(items), 4096)
        image = None
        for a in message.attachments:
            url = a.url
            if is_image(url) is not None:
                image = await self.data.exec.uproxy(url)
        if not image and message.embeds:
            for e in message.embeds:
                if e.image:
                    image = await self.data.exec.uproxy(e.image.url)
                if e.thumbnail:
                    image = await self.data.exec.uproxy(e.thumbnail.url)
        if image:
            emb.url = image
            emb.set_image(url=image)
        for e in message.embeds:
            if len(emb.fields) >= 25:
                break
            if not emb.description or emb.description == EmptyEmbed:
                title = e.title or ""
                if title:
                    emb.title = title
                emb.url = e.url or ""
                description = e.description or e.url or ""
                if description:
                    emb.description = description
            else:
                if e.title or e.description:
                    emb.add_field(name=e.title or e.url or "\u200b", value=lim_str(e.description, 1024) or e.url or "\u200b", inline=False)
            for f in e.fields:
                if len(emb.fields) >= 25:
                    break
                if f:
                    emb.add_field(name=f.name, value=f.value, inline=getattr(f, "inline", True))
            if len(emb) >= 6000:
                while len(emb) > 6000:
                    emb.remove_field(-1)
                break
        if not emb.description:
            urls = list(chain(("(" + e.url + ")" for e in message.embeds if e.url), ("[" + best_url(a) + "]" for a in message.attachments)))
            items = []
            for i in range((len(urls) + 9) // 10):
                temp = urls[i * 10:i * 10 + 10]
                temp2 = await self.data.exec.uproxy(*temp, collapse=False)
                items.extend(temp2[x] or temp[x] for x in range(len(temp)))
            emb.description = lim_str("\n".join(items), 4096)
        if link:
            emb.description = lim_str(f"{emb.description}\n\n[View Message](https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id})", 4096)
            emb.timestamp = message.edited_at or message.created_at
        return emb

    # Limits a cache to a certain amount, discarding oldest entries first.
    def limit_cache(self, cache=None, limit=None):
        if limit is None:
            limit = self.cache_size
        if cache is not None:
            caches = (self.cache[cache],)
        else:
            caches = self.cache.values()
        for c in caches:
            if len(c) >= limit * 2:
                items = tuple(c.items())
                c.clear()
                c.update(dict(items[-limit:]))

    def cache_reduce(self):
        self.limit_cache("messages")
        self.limit_cache("attachments", 256)
        # self.limit_cache("guilds")
        self.limit_cache("channels")
        self.limit_cache("users")
        self.limit_cache("members")
        self.limit_cache("roles")
        self.limit_cache("emojis")
        self.limit_cache("deleted", limit=16384)
        self.limit_cache("banned", 4096)

    # Updates bot cache from the discord.py client cache, using automatic feeding to mitigate the need for slow dict.update() operations.
    def update_cache_feed(self):
        self.cache.guilds._feed = (self._guilds, getattr(self, "sub_guilds", {}))
        self.cache.emojis._feed = (self._emojis,)
        self.cache.users._feed = (self._users,)
        g = self._guilds.values()
        self.cache.members._feed = lambda: (guild._members for guild in g)
        self.cache.channels._feed = lambda: chain(
            (guild._channels for guild in g),
            (guild._threads for guild in g),
            (self._private_channels,),
            (getattr(self, "sub_channels", {}),),
        )
        self.cache.roles._feed = lambda: (guild._roles for guild in g)

    def update_usernames(self):
        if self.users_updated:
            self.usernames = {str(user): user for user in self.cache.users.values()}

    def update_subs(self):
        self.sub_guilds = dict(self._guilds) or self.sub_guilds
        self.sub_channels = dict(chain.from_iterable(guild._channels.items() for guild in self.sub_guilds.values())) or self.sub_channels
        if not hasattr(self, "guilds_ready") or not self.guilds_ready.done():
            return
        for guild in self.guilds:
            if len(guild._members) != guild.member_count:
                print("Incorrect member count:", guild, len(guild._members), guild.member_count)
                create_task(self.load_guild_http(guild))

    # Gets the target bot prefix for the target guild, return the default one if none exists.
    def get_prefix(self, guild):
        try:
            g_id = guild.id
        except AttributeError:
            try:
                g_id = int(guild)
            except TypeError:
                g_id = 0
        with suppress(KeyError):
            return self.data.prefixes[g_id]
        return self.prefix

    # Gets effective permission level for the target user in a certain guild, taking into account roles.
    def get_perms(self, user, guild=None):
        try:
            u_id = user.id
        except AttributeError:
            u_id = int(user)
        if self.is_owner(u_id):
            return nan
        if self.is_blacklisted(u_id):
            return -inf
        if u_id == self.id:
            return inf
        if guild is None or hasattr(guild, "ghost"):
            return inf
        if u_id == guild.owner_id:
            return inf
        with suppress(KeyError):
            perm = self.data.perms[guild.id][u_id]
            if isnan(perm):
                return -inf
            return perm
        m = guild.get_member(u_id)
        if m is None:
            r = guild.get_role(u_id)
            if r is None:
                with suppress(KeyError):
                    return self.data.perms[guild.id][guild.id]
                return -inf
            return self.get_role_perms(r, guild)
        if m.bot:
            try:
                return self.data.perms[guild.id][guild.id]
            except KeyError:
                return 0
        try:
            p = m.guild_permissions
        except AttributeError:
            return -inf
        if p.administrator:
            return inf
        perm = -inf
        for role in m.roles:
            rp = self.get_role_perms(role, guild)
            if rp > perm:
                perm = rp
        if isnan(perm):
            perm = -inf
        return perm

    # Gets effective permission level for the target role in a certain guild, taking into account permission values.
    def get_role_perms(self, role, guild):
        if role.permissions.administrator:
            return inf
        with suppress(KeyError):
            perm = self.data.perms[guild.id][role.id]
            if isnan(perm):
                return -inf
            return perm
        if guild.id == role.id:
            return 0
        p = role.permissions
        if all((p.ban_members, p.manage_channels, p.manage_guild, p.manage_roles, p.manage_messages)):
            return 4
        elif any((p.ban_members, p.manage_channels, p.manage_guild)):
            return 3
        elif any((p.kick_members, p.manage_messages, p.manage_nicknames, p.manage_roles, p.manage_webhooks, p.manage_emojis)):
            return 2
        elif any((p.view_audit_log, p.priority_speaker, p.mention_everyone, p.move_members)):
            return 1
        return -1

    # Sets the permission value for a snowflake in a guild to a value.
    def set_perms(self, user, guild, value):
        perms = self.data.perms
        try:
            u_id = user.id
        except AttributeError:
            u_id = user
        g_perm = set_dict(perms, guild.id, {})
        g_perm.update({u_id: round_min(value)})
        self.data.perms.update()

    # Sets the permission value for a snowflake in a guild to a value.
    def remove_perms(self, user, guild):
        perms = self.data.perms
        try:
            u_id = user.id
        except AttributeError:
            u_id = user
        g_perm = set_dict(perms, guild.id, {})
        g_perm.pop(u_id, None)
        self.data.perms.update()

    def get_enabled(self, channel):
        guild = getattr(channel, "guild", None)
        if not guild and hasattr(channel, "member_count"):
            guild = channel
        if guild:
            try:
                enabled = self.data.enabled[channel.id]
            except KeyError:
                try:
                    enabled = self.data.enabled[guild.id]
                except KeyError:
                    enabled = ("main", "string", "admin", "voice", "image", "fun", "webhook")
        else:
            enabled = self.categories.keys()
        return enabled

    # Checks whether a member's status was changed.
    def status_changed(self, before, after):
        if before.activity != after.activity:
            return True
        for attr in ("status", "desktop_status", "web_status", "mobile_status"):
            b, a = getattr(before, attr), getattr(after, attr)
            if b == a:
                return False
        return True

    # Checks whether a member's status was updated by themselves.
    def status_updated(self, before, after):
        if before.activity != after.activity:
            return True
        for attr in ("status", "desktop_status", "web_status", "mobile_status"):
            b, a = getattr(before, attr), getattr(after, attr)
            if b == discord.Status.online and a == discord.Status.idle:
                if utc() - self.data.users.get(after.id, {}).get("last_seen", 0) < 900:
                    return False
            elif a == discord.Status.offline:
                return False
            elif b == a:
                return False
        return True

    # Checks if a message has been flagged as deleted by the deleted cache.
    def is_deleted(self, message):
        try:
            m_id = int(message.id)
        except AttributeError:
            m_id = int(message)
        return self.cache.deleted.get(m_id, False)

    # Logs if a message has been deleted.
    def log_delete(self, message, no_log=False):
        if not message:
            return
        try:
            m_id = int(message.id)
        except AttributeError:
            m_id = int(message)
        self.cache.deleted[m_id] = no_log + 2

    # Silently deletes a message, bypassing logs.
    async def silent_delete(self, message, exc=False, no_log=False, delay=None):
        if not message:
            return
        if type(message) is int:
            message = await self.fetch_message(message)
        if delay:
            await asyncio.sleep(float(delay))
        try:
            self.log_delete(message, no_log)
            await discord.Message.delete(message)
        except:
            self.cache.deleted.pop(message.id, None)
            if exc:
                raise

    async def verified_ban(self, user, guild, reason=None):
        self.cache.banned[(guild.id, user.id)] = utc()
        try:
            await guild.ban(user, delete_message_days=0, reason=reason)
        except:
            self.cache.banned.pop((guild.id, user.id), None)
            raise
        self.cache.banned[(guild.id, user.id)] = utc()

    def recently_banned(self, user, guild, duration=20):
        return utc() - self.cache.banned.get((verify_id(guild), verify_id(user)), 0) < duration

    def is_mentioned(self, message, user, guild=None):
        u_id = verify_id(user)
        if u_id in (member.id for member in message.mentions):
            return True
        if guild is None:
            return True
        if "exec" in self.data and self.data.exec.get(message.channel.id, 0) & 64 and (not message.content or no_md(message.content[0]) not in "\\#"):
            return True
        member = guild.get_member(u_id)
        if member is None:
            return False
        if message.content.count("`") > 1:
            return False
        for role in member.roles:
            if not role.mentionable:
                if role.mention in message.content:
                    return True
        return False

    # Checks if a user is an owner of the bot.
    is_owner = lambda self, user: verify_id(user) in self.owners

    # Checks if a guild is trusted by the bot.
    def is_trusted(self, guild):
        try:
            trusted = self.data.trusted
        except (AttributeError, KeyError):
            return 0
        if not guild:
            return 0
        i = verify_id(guild)
        if i not in trusted:
            return 0
        if not isinstance(trusted[i], set):
            try:
                trusted[i] = set(trusted[i])
            except TypeError:
                trusted[i] = set()
        for u in tuple(trusted[i]):
            if u is None:
                continue
            if u in self.data.premiums and self.data.premiums[u]["lv"] >= 2:
                pass
            else:
                print(i, "trusted lost from", u)
                trusted[i].remove(u)
                trusted.update(i)
        trusted[i].add(None)
        return min(2, len(trusted[i]))

    # Checks a user's premium subscription level.
    def premium_level(self, user, absolute=False):
        try:
            premiums = self.data.premiums
        except (AttributeError, KeyError):
            return 0
        uid = verify_id(user)
        lv = 0
        if self.premium_server in self.cache.guilds:
            u = self.cache.guilds[self.premium_server].get_member(uid)
            if u:
                for role in u.roles:
                    if role.id in self.premium_roles:
                        lv = max(lv, self.premium_roles[role.id])
        if not absolute:
            data = bot.data.users.get(uid)
            if data and data.get("trial"):
                if lv >= 2:
                    data.pop("trial")
                    bot.data.users.update(uid)
                elif data.get("diamonds", 0) >= 1:
                    lv = max(lv, data["trial"])
                else:
                    data.pop("trial")
                    bot.data.users.update(uid)
            premiums.subscribe(user, lv)
        return lv

    def is_nsfw(self, channel):
        if is_nsfw(channel):
            return True
        if "nsfw" in self.data and getattr(channel, "recipient", None):
            return self.data.nsfw.get(channel.recipient.id, False)

    async def donate(self, name, uid, amount, msg):
        channel = self.get_channel(320915703102177293)
        if not channel:
            return
        if msg:
            emb = discord.Embed(colour=rand_colour())
            emb.set_author(**get_author(self.user))
            emb.description = msg
        else:
            emb = None
        if not uid:
            await channel.send(f"Failed to locate donation of ${amount} from user {name}!", embed=emb)
            return
        await create_future(self.update_usernames)
        try:
            user = await self.fetch_user(uid)
        except:
            print_exc()
            await channel.send(f"Failed to locate donation of ${amount} from user {name}/{uid}!", embed=emb)
            return
        dias = round_min(amount * 300)
        self.data.users.add_diamonds(user, dias, multiplier=False)
        create_task(channel.send(f"Thank you {user_mention(user.id)} for donating ${amount}! Your account has been credited 💎 {dias}!", embed=emb))
        await user.send(f"Thank you for donating ${amount}! Your account has been credited 💎 {dias}!")
        return True

    # Checks if a user is blacklisted from the bot.
    def is_blacklisted(self, user):
        u_id = verify_id(user)
        if self.is_owner(u_id) or u_id == self.id:
            return False
        with suppress(KeyError):
            return (self.data.blacklist.get(u_id) or 0) > 1
        return True

    dangerous_command = bold(css_md(uni_str('[WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. REPEAT COMMAND WITH "?f" FLAG TO CONFIRM.]'), force=True))

    mmap = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "〝": '"',
        "〞": '"',
        "⸌": "'",
        "⸍": "'",
        "⸢": "'",
        "⸣": "'",
        "⸤": "'",
        "⸥": "'",
        "⸨": "((",
        "⸩": "))",
        "⟦": "[",
        "⟧": "]",
        "〚": "[",
        "〛": "]",
        "「": "[",
        "」": "]",
        "『": "[",
        "』": "]",
        "【": "[",
        "】": "]",
        "〖": "[",
        "〗": "]",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "⌈": "[",
        "⌉": "]",
        "⌊": "[",
        "⌋": "]",
        "⦋": "[",
        "⦌": "]",
        "⦍": "[",
        "⦐": "]",
        "⦏": "[",
        "⦎": "]",
        "⁅": "[",
        "⁆": "]",
        "〔": "[",
        "〕": "]",
        "«": "<<",
        "»": ">>",
        "❮": "<",
        "❯": ">",
        "❰": "<",
        "❱": ">",
        "❬": "<",
        "❭": ">",
        "＜": "<",
        "＞": ">",
        "⟨": "<",
        "⟩": ">",
    }
    mtrans = "".maketrans(mmap)

    cmap = {
        "<": "alist((",
        ">": "))",
    }
    ctrans = "".maketrans(cmap)

    op = {
        "=": None,
        ":=": None,
        "+=": "__add__",
        "-=": "__sub__",
        "*=": "__mul__",
        "/=": "__truediv__",
        "//=": "__floordiv__",
        "**=": "__pow__",
        "^=": "__pow__",
        "%=": "__mod__",
    }
    consts = {
        "k": 1 << 10,
        "M": 1 << 20,
        "G": 1 << 30,
        "T": 1 << 40,
        "P": 1 << 50,
        "E": 1 << 60,
        "Z": 1 << 70,
        "Y": 1 << 80,
    }

    # Evaluates a math formula to a float value, using a math process from the subprocess pool when necessary.
    async def eval_math(self, expr, default=0, op=True):
        expr = as_str(expr)
        if op:
            # Allow mathematical operations on a default value
            _op = None
            for op, at in self.op.items():
                if expr.startswith(op):
                    expr = expr[len(op):].strip()
                    _op = at
            num = await self.eval_math(expr, op=False)
            if _op is not None:
                num = getattr(mpf(default), _op)(num)
            return num
        f = expr.strip()
        try:
            if not f:
                return 0
            else:
                s = f.casefold()
                if len(s) < 5 and s in ("t", "true", "y", "yes", "on"):
                    return True
                elif len(s) < 6 and s in ("f", "false", "n", "no", "off"):
                    return False
                else:
                    try:
                        return round_min(f)
                    except:
                        try:
                            return orjson.loads(f)
                        except:
                            return ast.literal_eval(f)
        except (ValueError, TypeError, SyntaxError):
            r = await self.solve_math(f, 128, 0, variables=self.consts)
        x = r[0]
        with suppress(TypeError):
            while True:
                if type(x) is str:
                    raise TypeError
                x = tuple(x)[0]
        if type(x) is str and x.isnumeric():
            return int(x)
        if type(x) is float:
            return x
        x = round_min(x)
        if type(x) is not int and len(str(x)) <= 16:
            return float(x)
        return x

    # Evaluates a math formula to a list of answers, using a math process from the subprocess pool when necessary.
    def solve_math(self, f, prec=128, r=False, timeout=16, variables=None):
        return process_math(f.strip(), int(prec), int(r), timeout=timeout, variables=variables)

    TimeChecks = {
        "galactic years": ("gy", "galactic year", "galactic years"),
        "millennia": ("ml", "millenium", "millenia"),
        "centuries": ("c", "century", "centuries"),
        "decades": ("dc", "decade", "decades"),
        "years": ("y", "year", "years"),
        "months": ("mo", "mth", "month", "mos", "mths", "months"),
        "weeks": ("w", "wk", "week", "wks", "weeks"),
        "days": ("d", "day", "days"),
        "hours": ("h", "hr", "hour", "hrs", "hours"),
        "minutes": ("m", "min", "minute", "mins", "minutes"),
        "seconds": ("s", "sec", "second", "secs", "seconds"),
    }
    num_words = "(?:(?:(?:[0-9]+|[a-z]{1,}illion)|thousand|hundred|ten|eleven|twelve|(?:thir|four|fif|six|seven|eigh|nine)teen|(?:twen|thir|for|fif|six|seven|eigh|nine)ty|zero|one|two|three|four|five|six|seven|eight|nine)\\s*)"
    numericals = re.compile("^(?:" + num_words + "|(?:a|an)\\s*)(?:" + num_words + ")*", re.I)
    connectors = re.compile("\\s(?:and|at)\\s", re.I)
    alphabet = frozenset("abcdefghijklmnopqrstuvwxyz")

    # Evaluates a time input, using a math process from the subprocess pool when necessary.
    async def eval_time(self, expr, default=0, op=True):
        if op:
            # Allow mathematical operations on a default value
            _op = None
            for op, at in self.op.items():
                if expr.startswith(op):
                    expr = expr[len(op):].strip(" ")
                    _op = at
            num = await self.eval_time(expr, op=False)
            if _op is not None:
                num = getattr(float(default), _op)(num)
            return num
        t = 0
        if expr:
            f = None
            if " " in expr:
                # Parse timezones first
                args = smart_split(expr)
                for a in (args[0], args[-1]):
                    tz = a.casefold()
                    if tz in TIMEZONES:
                        t = -get_timezone(tz)
                        expr = expr.replace(a, "")
                        break
            day = None
            try:
                # Try to evaluate time inputs
                if ":" in expr:
                    data = expr.split(":")
                    mult = 1
                    while len(data):
                        t += await self.eval_math(data.pop(-1)) * mult
                        if mult <= 60:
                            mult *= 60
                        elif mult <= 3600:
                            mult *= 24
                        elif len(data):
                            raise TypeError("Too many time arguments.")
                else:
                    try:
                        t = float(expr)
                    except:
                        # Otherwise move on to main parser
                        f = single_space(self.connectors.sub(" ", expr.replace(",", " "))).casefold()
                        if "today" in f:
                            day = 0
                            f = f.replace("today", "")
                        elif "tomorrow" in f:
                            day = 1
                            f = f.replace("tomorrow", "")
                        elif "yesterday" in f:
                            day = -1
                            f = f.replace("yesterday", "")
                        if day is not None:
                            raise StopIteration
                        dd = {}
                        td = {}
                        for tc in self.TimeChecks:
                            for check in reversed(self.TimeChecks[tc]):
                                if check in f:
                                    i = f.index(check)
                                    isnt = i + len(check) < len(f) and f[i + len(check)] in self.alphabet
                                    if isnt or not i or f[i - 1] in self.alphabet:
                                        continue
                                    temp = f[:i]
                                    f = f[i + len(check):].strip()
                                    match = self.numericals.search(temp)
                                    if match:
                                        i = match.end()
                                        n = num_parse(temp[:i])
                                        temp = temp[i:].strip()
                                        if temp:
                                            f = f"{temp} {f}"
                                    else:
                                        n = await self.eval_math(temp)
                                    if tc == "weeks":
                                        add_dict(td, {"days": n * 7})
                                    elif tc in ("days", "hours", "minutes", "seconds"):
                                        add_dict(td, {tc: n})
                                    else:
                                        add_dict(dd, {tc: n})
                        temp = f.strip()
                        if temp:
                            match = self.numericals.search(temp)
                            if match:
                                i = match.end()
                                n = num_parse(temp[:i])
                                temp = temp[i:].strip()
                                if temp:
                                    n = await self.eval_math(f"{n} {temp}")
                            else:
                                n = await self.eval_math(temp)
                            t += n
                        t += td.get("seconds", 0)
                        t += td.get("minutes", 0) * 60
                        t += td.get("hours", 0) * 3600
                        t += td.get("days", 0) * 86400
                        if dd:
                            ts = utc()
                            dt = utc_dft(t + ts)
                            years = dd.get("years", 0) + dd.get("decades", 0) * 10 + dd.get("centuries", 0) * 100 + dd.get("millennia", 0) * 1000 + dd.get("galactic years", 0) * 226814
                            dt = dt.add_years(years)
                            months = dd.get("months", 0)
                            dt = dt.add_months(months)
                            t = dt.timestamp() - ts
            except:
                # Use datetime parser if regular parser fails
                raw = tzparse(f if f else expr)
                if day is not None:
                    curr = utc() + day * 86400
                    while raw < curr:
                        raw += 86400
                    while raw - curr > 86400:
                        raw -= 86400
                t = (raw - utc_ddt()).total_seconds()
        if not isinstance(t, (int, float)):
            try:
                t = float(t)
            except OverflowError:
                t = int(t)
        return t

    # Updates the bot's stored external IP address.
    def update_ip(self, ip):
        if regexp("^([0-9]{1,3}\\.){3}[0-9]{1,3}$").search(ip):
            self.ip = ip
            new_ip = f"http://{self.ip}:{PORT}"
            # if self.raw_webserver != self.webserver and self.raw_webserver != new_ip:
            #     create_task(self.create_main_website())
            # self.raw_webserver = new_ip

    def is_webserver_url(self, url):
        if url.startswith(self.raw_webserver) or url.startswith("https://" + self.raw_webserver.split("//", 1)[-1]):
            return (url,)
        return regexp("^https?:\\/\\/(?:[A-Za-z]+\\.)?mizabot\\.xyz").findall(url)

    # Gets the external IP address from api.ipify.org
    async def get_ip(self):
        self.ip = await Request("https://api.ipify.org", bypass=False, decode=True, aio=True)
        return self.ip
        # self.update_ip(resp)

    # Gets the amount of active processes, threads, coroutines.
    def get_active(self):
        procs = 2 + sum(1 for c in self.proc.children(True))
        thrds = self.proc.num_threads()
        coros = sum(1 for i in asyncio.all_tasks(self.loop))
        return alist((procs, thrds, coros))

    # Gets the CPU and memory usage of a process over a period of 1 second.
    async def get_proc_state(self, proc):
        with suppress(psutil.NoSuchProcess):
            c = await create_future(proc.cpu_percent, priority=True)
            if not c:
                await asyncio.sleep(1)
                c = await create_future(proc.cpu_percent)
            # m = proc.memory_percent()
            m = proc.memory_info().vms
            return float(c), float(m)
        return 0, 0

    async def get_disk(self):
        with tracebacksuppressor(SemaphoreOverflowError, FileNotFoundError):
            async with self.disk_semaphore:
                self.file_count, self.disk = await create_future(get_folder_size, ".", priority=True)
        return self.disk

    async def get_hosted(self):
        self.__dict__.setdefault("total_hosted", 0)
        size = 0
        for fn in os.listdir("saves/filehost"):
            with tracebacksuppressor(ValueError):
                if "$" in fn and fn.split("$", 1)[0].endswith("~.forward"):
                    size += int(fn.split("$", 2)[1])
                else:
                    p = "saves/filehost/" + fn
                    size += os.path.getsize(p)
        self.total_hosted = size
        return size

    # Gets the status of the bot.
    @tracebacksuppressor
    async def get_state(self):
        stats = azero(3)
        procs = await create_future(self.proc.children, recursive=True, priority=True)
        procs.append(self.proc)
        tasks = [self.get_proc_state(p) for p in procs]
        resp = await recursive_coro(tasks)
        stats += [sum(st[0] for st in resp), sum(st[1] for st in resp), 0]
        cpu = psutil.cpu_count(logical=True)
        # mem = psutil.virtual_memory()
        # CPU is totalled across all cores
        stats[0] /= cpu
        # Memory is in %
        # stats[1] *= mem.total / 100
        stats[2] = self.disk
        self.size2 = fcdict()
        files = os.listdir("misc")
        for f in files:
            path = "misc/" + f
            if is_code(path):
                self.size2[f] = line_count(path)
        self.curr_state = stats
        return stats

    status_sem = Semaphore(1, inf, rate_limit=1)

    def status(self):
        if self.status_sem.busy:
            self.status_sem.wait()
            return self.status_data
        with self.status_sem:
            self.status_data = {}
            active = self.get_active()
            size = (
                np.sum(deque(self.size.values()), dtype=np.uint32, axis=0)
                + np.sum(deque(self.size2.values()), dtype=np.uint32, axis=0)
            )
            stats = self.curr_state
            commands = set(itertools.chain(*self.commands.values()))
            try:
                audio_players = len(self.audio.players)
            except:
                audio_players = active_audio_players = "N/A"
            else:
                active_audio_players = sum(bool(auds.queue and not auds.paused) for auds in self.audio.players.values())
            self.status_data = {
                "System info": {
                    "Process count": active[0],
                    "Thread count": active[1],
                    "Coroutine count": active[2],
                    "CPU usage": f"{round(stats[0], 3)}%",
                    "RAM usage": byte_scale(stats[1]) + "B",
                    "Disk usage": byte_scale(stats[2]) + "B",
                    "Network usage": byte_scale(bot.bitrate) + "bps",
                    "Stored files": self.file_count,
                },
                "Discord info": {
                    "Shard count": self.shards,
                    "Server count": len(self._guilds),
                    "User count": len(self.cache.users),
                    "Channel count": len(self.cache.channels),
                    "Role count": len(self.cache.roles),
                    "Emoji count": len(self.cache.emojis),
                    "Cached messages": len(self.cache.messages),
                    "API latency": sec2time(self.api_latency),
                },
                "Misc info": {
                    "Active commands": self.command_semaphore.active,
                    "Connected voice channels": audio_players,
                    "Active audio players": active_audio_players,
                    "Activity count": self.activity,
                    "Total data transmitted": byte_scale(bot.total_bytes) + "B",
                    "Hosted storage": byte_scale(bot.total_hosted) + "B",
                    "System time": datetime.datetime.now(),
                    "Current uptime": dyn_time_diff(utc(), bot.start_time),
                },
                "Code info": {
                    "Code size": [x.item() for x in size],
                    "Command count": len(commands),
                    "Website URL": self.webserver,
                },
            }
        return self.status_data

    # Loads a module containing commands and databases by name.
    @tracebacksuppressor
    def get_module(self, module):
        f = module
        if "." in f:
            f = f[:f.rindex(".")]
        path, module = module, f
        new = False
        reloaded = False
        if module in self._globals:
            reloaded = True
            print(f"Reloading module {module}...")
            if module in self.categories:
                self.unload(module)
            # mod = importlib.reload(self._globals[module])
        else:
            print(f"Loading module {module}...")
            new = True
            # mod = __import__(module)
        if not new:
            mod = self._globals.pop(module, {})
        # else:
        #     mod = self._globals
        else:
            mod = cdict(common.__dict__)
        fn = f"commands/{module}.py"
        with open(fn, "rb") as f:
            b = f.read()
        code = compile(b, fn, "exec", optimize=1)
        exec(code, mod)
        self._globals[module] = mod
        commands = deque()
        dataitems = deque()
        items = mod
        for var in tuple(items.values()):
            if callable(var) and var is not Command and var is not Database:
                load_type = 0
                with suppress(TypeError):
                    if issubclass(var, Command):
                        load_type = 1
                    elif issubclass(var, Database) and not reloaded:
                        load_type = 2
                if load_type:
                    obj = var(self, module)
                    if load_type == 1:
                        commands.append(obj)
                    elif load_type == 2:
                        dataitems.append(obj)
        commands = alist(commands)
        dataitems = alist(dataitems)
        for u in dataitems:
            for c in commands:
                c.data[u.name] = u
        self.categories[module] = commands
        self.dbitems[module] = dataitems
        self.size[module] = line_count("commands/" + path)
        if commands:
            print(f"{module}: Successfully loaded {len(commands)} command{'s' if len(commands) != 1 else ''}.")
        if dataitems:
            print(f"{module}: Successfully loaded {len(dataitems)} database{'s' if len(dataitems) != 1 else ''}.")
        if not new and not reloaded:
            while not self.ready:
                time.sleep(0.5)
            print(f"Resending _ready_ event to module {module}...")
            for db in dataitems:
                for f in dir(db):
                    if f.startswith("_") and f[-1] == "_" and f[1] != "_":
                        func = getattr(db, f, None)
                        if callable(func):
                            self.events.append(f, func)
                for e in ("_bot_ready_", "_ready_"):
                    func = getattr(db, e, None)
                    if callable(func):
                        await_fut(create_future(func, bot=self))
        print(f"Successfully loaded module {module}.")
        return True

    @tracebacksuppressor
    def unload(self, mod=None):
        if mod is None:
            mods = deque(self.categories)
        else:
            mod = mod.casefold()
            if mod not in self.categories:
                raise KeyError
            mods = [mod]
        for mod in mods:
            for command in self.categories[mod]:
                command.unload()
            # for database in self.dbitems[mod]:
            #     database.unload()
            self.categories.pop(mod)
            # self.dbitems.pop(mod)
            self.size.pop(mod)
        return True

    def reload(self, mod=None):
        sub_kill()
        if not mod:
            modload = deque()
            files = [i for i in os.listdir("commands") if is_code(i) and i.rsplit(".", 1)[0] in self.active_categories]
            for f in files:
                modload.append(create_future_ex(self.get_module, f, priority=True))
            create_future_ex(self.start_audio_client)
            create_task(self.create_main_website())
            return all(fut.result() for fut in modload)
        create_task(self.create_main_website())
        return self.get_module(mod + ".py")

    # Loads all modules in the commands folder and initializes bot commands and databases.
    def get_modules(self):
        files = [i for i in os.listdir("commands") if is_code(i) and i.rsplit(".", 1)[0] in self.active_categories]
        self.categories = fcdict()
        self.dbitems = fcdict()
        self.commands = fcdict()
        self.data = fcdict()
        self.database = fcdict()
        self.size = fcdict()
        for f in os.listdir():
            if is_code(f):
                self.size[f] = line_count(f)
        self.modload = deque()
        for f in files:
            self.modload.append(create_future(self.get_module, f, priority=True))
        self.loaded = True

    def clear_cache(self):
        if "audio" in self.data:
            if self._globals["VOICE"].ytdl.download_sem.active:
                return 0
        if self.cache_semaphore.busy:
            return 0
        with self.cache_semaphore:
            i = 0
            expendable = list(os.scandir("cache"))
            stats = psutil.disk_usage(os.getcwd())
            t = utc()
            expendable = sorted(expendable, key=lambda f: ((t - max(f.stat().st_atime, f.stat().st_mtime)) // 3600, f.stat().st_size), reverse=True)
            if not expendable:
                return 0
            while stats.free < 81 * 1073741824 or len(expendable) > 4096 or (t - expendable[0].stat().st_atime) > 3600 * 12:
                if not expendable:
                    break
                with tracebacksuppressor:
                    os.remove(expendable.pop(0).path)
                    i += 1
            if i > 1:
                print(f"{i} cached files flagged for deletion.")
            return i

    @tracebacksuppressor
    def backup(self):
        backup = AUTH.get("backup_path") or "backup"
        self.clear_cache()
        date = datetime.datetime.utcnow().date()
        fn = f"{backup}/saves.{date}.wb"
        if os.path.exists(fn):
            if utc() - os.path.getmtime(fn) < 60:
                return fn
            os.remove(fn)
        for i in range(30):
            d2 = date - datetime.timedelta(days=i + 3)
            f2 = f"{backup}/saves.{d2}.wb"
            if os.path.exists(f2):
                os.remove(f2)
                continue
            break
        lines = as_str(subprocess.run([sys.executable, "neutrino.py", "-c0", "../saves", "../" + fn], stderr=subprocess.PIPE, cwd="misc").stdout).splitlines()
        s = "\n".join(line for line in lines if not line.startswith("\r"))
        print(s)
        # zf = ZipFile(fn, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True)
        # for x, y, z in os.walk("saves"):
        #     for f in z:
        #         fp = os.path.join(x, f)
        #         zf.write(fp, fp)
        # zf.close()
        print("Backup database created in", fn)
        return fn

    # Autosaves modified bot databases. Called once every minute and whenever the bot is about to shut down.
    def update(self, force=False):
        if force:
            self.update_embeds(True)
        saved = alist()
        with tracebacksuppressor:
            for i, u in self.data.items():
                if getattr(u, "update", None):
                    with MemoryTimer(str(u)):
                        if u.update(force=True):
                            saved.append(i)
                            time.sleep(0.05)
        if not self.closed and hasattr(self, "total_bytes"):
            with tracebacksuppressor:
                s = "{'net_bytes': " + str(self.total_bytes) + "}"
                saves = "saves/status.json"
                if os.path.exists("saves"):
                    with open(saves[:-5] + "\x7f.json", "w") as f:
                        f.write(s)
                    with suppress(FileNotFoundError):
                        os.remove(saves[:-5] + "\x7f\x7f.json")
                    with suppress(FileNotFoundError):
                        os.rename(saves, saves[:-5] + "\x7f\x7f.json")
                    os.rename(saves[:-5] + "\x7f.json", saves)
                with open(saves, "w") as f:
                    f.write(s)
        backup = AUTH.get("backup_path") or "backup"
        if not os.path.exists(backup):
            os.mkdir(backup)
        fn = f"{backup}/saves.{datetime.datetime.utcnow().date()}.wb"
        day = not os.path.exists(fn)
        if day:
            await_fut(self.send_event("_day_"))
            self.users_updated = True
        if force or day:
            fut = self.send_event("_save_")
            await_fut(fut)
        if day:
            self.backup()
            futs = deque()
            das = deque()
            for u in self.data.values():
                if not xrand(5) and not u._garbage_semaphore.busy:
                    futs.append(create_task(self.garbage_collect(u)))
                    das.append(u)
            for fut in futs:
                await_fut(fut)
            for u in das:
                u.vacuum()

    async def as_rewards(self, diamonds, gold=Dummy):
        if type(diamonds) is not int:
            diamonds = floor(diamonds)
        if type(gold) is not int:
            gold = floor(gold)
        if gold is Dummy:
            gold = diamonds
            diamonds = 0
        out = deque()
        if diamonds:
            out.append(f"💎 {diamonds}")
        if gold:
            coin = await create_future(self.data.emojis.emoji_as, "miza_coin.gif")
            out.append(f"{coin} {gold}")
        if out:
            return " ".join(out)
        return

    zw_callback = zwencode("callback")

    # Operates on reactions on special messages, calling the _callback_ methods of commands when necessary.
    async def react_callback(self, message, reaction, user):
        if message.author.id == self.id:
            if self.closed:
                return
            u_perm = self.get_perms(user.id, message.guild)
            if u_perm <= -inf:
                return
            if reaction is not None:
                reacode = str(reaction).encode("utf-8")
            else:
                reacode = None
            m = self.cache.messages.get(message.id)
            if getattr(m, "_react_callback_", None):
                await m._react_callback_(
                    message=message,
                    channel=message.channel,
                    guild=message.guild,
                    reaction=reacode,
                    user=user,
                    perm=u_perm,
                    vals="",
                    argv="",
                    bot=self,
                )
                # await self.send_event("_callback_", user=user, command=f, loop=False, message=message)
                return
            msg = message.content.strip("*")
            if not msg and message.embeds:
                msg = str(message.embeds[0].description).strip("*")
            if msg[:3] != "```" or len(msg) <= 3:
                msg = None
                if message.embeds:
                    s = message.embeds[0].footer.text
                    if is_zero_enc(s):
                        msg = s
                if not msg:
                    return
            else:
                msg = msg[3:].lstrip("\n")
                check = "callback-"
                with suppress(ValueError):
                    msg = msg[:msg.index("\n")]
                if not msg.startswith(check):
                    return
            while len(self.react_sem) > 65536:
                with suppress(RuntimeError):
                    self.react_sem.pop(next(iter(self.react_sem)))
            while utc() - self.react_sem.get(message.id, 0) < 30:
                # Ignore if more than 2 reactions already queued for target message
                if self.react_sem.get(message.id, 0) - utc() > 1:
                    return
                await asyncio.sleep(0.2)
            msg = message.content.strip("*")
            if not msg and message.embeds:
                msg = str(message.embeds[0].description).strip("*")
            if msg[:3] != "```" or len(msg) <= 3:
                msg = None
                if message.embeds:
                    s = message.embeds[0].footer.text
                    if is_zero_enc(s):
                        msg = s
                if not msg:
                    return
                # Experimental zero-width invisible character encoded message (unused)
                try:
                    msg = msg[msg.index(self.zw_callback) + len(self.zw_callback):]
                except ValueError:
                    return
                msg = zwdecode(msg)
                args = msg.split("q")
            else:
                msg = msg[3:].lstrip("\n")
                check = "callback-"
                msg = msg.splitlines()[0]
                msg = msg[len(check):]
                args = msg.split("-")
            catn, func, vals = args[:3]
            func = func.casefold()
            argv = "-".join(args[3:])
            catg = self.categories[catn]
            # Force a rate limit on the reaction processing for the message
            self.react_sem[message.id] = max(utc(), self.react_sem.get(message.id, 0) + 1)
            for f in catg:
                if f.parse_name().casefold() == func:
                    with self.ExceptionSender(message.channel, reference=message):
                        timeout = getattr(f, "_timeout_", 1) * self.timeout
                        if timeout >= inf:
                            timeout = None
                        elif self.is_trusted(message.guild):
                            timeout *= 3
                        await asyncio.wait_for(
                            f._callback_(
                                message=message,
                                channel=message.channel,
                                guild=message.guild,
                                reaction=reacode,
                                user=user,
                                perm=u_perm,
                                vals=vals,
                                argv=argv,
                                bot=self,
                            ),
                            timeout=timeout)
                        await self.send_event("_command_", user=user, command=f, loop=False, message=message)
                        break
            self.react_sem.pop(message.id, None)

    @tracebacksuppressor
    async def update_status(self):
        guild_count = len(self.guilds)
        changed = guild_count != self.guild_count
        if changed or utc() > self.stat_timer:
            self.stat_timer = utc() + 4.5
            self.guild_count = guild_count
            status_changes = list(range(self.status_iter))
            status_changes.extend(range(self.status_iter + 1, len(self.statuses) - (not self.audio)))
            if not status_changes:
                status_changes = range(len(self.statuses))
            self.status_iter = choice(status_changes)
            with suppress(discord.NotFound):
                if "blacklist" in self.data and self.data.blacklist.get(0):
                    text = "Currently under maintenance, please stay tuned!"
                else:
                    text = f"{self.webserver}, to {uni_str(guild_count)} server{'s' if guild_count != 1 else ''}"
                    if self.owners:
                        u = await self.fetch_user(next(iter(self.owners)))
                        n = u.name
                        text += f", from {belongs(uni_str(n))} place!"
                    else:
                        text += "!"
                # Status iterates through 5 possible choices
                status = self.statuses[self.status_iter]
                if status is discord.Streaming:
                    activity = discord.Streaming(name=text, url=self.twitch_url)
                    status = discord.Status.dnd
                else:
                    activity = discord.Game(name=text)
                if changed:
                    print(repr(activity))
                if self.audio:
                    audio_status = f"await client.change_presence(status=discord.Status."
                    if status == discord.Status.invisible:
                        status = discord.Status.idle
                        create_task(self.audio.asubmit(audio_status + "online)"))
                        await self.seen(self.user, event="misc", raw="Changing their status")
                    else:
                        if status == discord.Status.online:
                            create_task(self.audio.asubmit(audio_status + "dnd)"))
                        create_task(self.seen(self.user, event="misc", raw="Changing their status"))
                with suppress(ConnectionResetError):
                    await self.change_presence(activity=activity, status=status)
                # Member update events are not sent through for the current user, so manually send a _seen_ event
                await self.seen(self.user, event="misc", raw="Changing their status")

    # Handles all updates to the bot. Manages the bot's status and activity on discord, and updates all databases.
    async def handle_update(self, force=False):
        if utc() - self.last_check > 5 or force:
            semaphore = self.semaphore if not force else emptyctx
            with suppress(SemaphoreOverflowError):
                with semaphore:
                    self.last_check = utc()
                    if not force:
                        create_task(self.get_state())
                    if self.bot_ready:
                        # Update databases
                        for u in self.data.values():
                            if not u._semaphore.busy:
                                trace(create_future(u, priority=True))

    # Processes a message, runs all necessary commands and bot events. May be called from another source.
    async def process_message(self, message, msg=None, edit=True, orig=None, loop=False, slash=False):
        if self.closed:
            return 0
        msg = msg if msg is not None else message.content
        cpy = msg
        # Get user, channel, guild that the message belongs to
        user = message.author
        guild = message.guild
        u_id = user.id
        channel = message.channel
        c_id = channel.id
        if guild:
            g_id = guild.id
        else:
            g_id = 0
        if not slash:
            if u_id != self.id:
                # Strip quote from message.
                if msg[:2] == "> ":
                    msg = msg[2:]
                # Strip spoiler from message.
                elif msg[:2] == "||" and msg[-2:] == "||":
                    msg = msg[2:-2]
                # Strip code boxes from message.
                msg = msg.replace("`", "").strip()
        # Get list of enabled commands for the channel.
        enabled = self.get_enabled(channel)
        u_perm = self.get_perms(u_id, guild)
        admin = not inf > u_perm
        # Gets prefix for current guild.
        if u_id == self.id:
            prefix = self.prefix
        else:
            prefix = self.get_prefix(guild)
        if not slash:
            op = False
            comm = msg
            # Mentioning the bot serves as an alias for the prefix.
            for check in self.mention:
                if comm.startswith(check):
                    prefix = self.prefix
                    comm = comm[len(check):].strip()
                    op = ("Unintentional command?", f"If you meant to chat with me instead, use {prefix}ask or one of its aliases to avoid accidentally triggering a command in the future!")
                    break
            if comm.startswith(prefix):
                comm = comm[len(prefix):].strip()
                op = True
        else:
            comm = msg
            if comm and (comm[0] == "/" or comm[0] == self.prefix):
                comm = comm[1:]
            op = True
        mentioning = (op or self.id in (member.id for member in message.mentions))
        if (op or mentioning) and "blacklist" in self.data and not isnan(u_perm):
            gid = self.data.blacklist.get(0)
            if gid and gid != g_id:
                create_task(send_with_react(
                    channel,
                    "I am currently under maintenance, please stay tuned!",
                    reacts="❎",
                    reference=message,
                ))
                return 0
        # Respond to blacklisted users attempting to use a command, or when mentioned without a command.
        if (u_perm <= -inf and mentioning) and not cpy.startswith("~~"):
            # print(f"Ignoring command from blacklisted user {user} ({u_id}): {lim_str(message.content, 256)}")
            if not self.ready:
                create_task(send_with_react(
                    channel,
                    "I am currently in the process of restarting, please hold tight!",
                    reacts="❎",
                    reference=message,
                ))
            else:
                create_task(send_with_react(
                    channel,
                    "Sorry, you are currently not permitted to request my services.",
                    reacts="❎",
                    reference=message,
                ))
            return 0
        truemention = True
        if self.id in (member.id for member in message.mentions):
            if message.reference:
                mid = getattr(message.reference, "message_id", None) or getattr(message.reference, "id", None)
                try:
                    m = await self.fetch_message(mid, message.channel)
                except:
                    pass
                else:
                    truemention = m.author.id != self.id and all(s not in message.content for s in self.mention)
            if truemention:
                try:
                    await self.send_event("_mention_", user=user, message=message, msg=msg, exc=True)
                except CommandCancelledError:
                    return 0
        remaining = 0
        run = False
        if op:
            # Special case: the ? alias for the ~help command, since ? is an argument flag indicator and will otherwise be parsed as one.
            if len(comm) and comm[0] == "?":
                command_check = comm[0]
                i = 1
            else:
                # Parse message to find command.
                i = len(comm)
                for end in " ?-+\t\n":
                    with suppress(ValueError):
                        i2 = comm.index(end)
                        if i2 < i:
                            i = i2
                command_check = full_prune(comm[:i]).replace("*", "").replace("_", "").replace("||", "")
            # Hash table lookup for target command: O(1) average time complexity.
            if command_check in bot.commands:
                # Multiple commands may have the same alias, run all of them
                for command in bot.commands[command_check]:
                    # Make sure command is enabled, administrators bypass this
                    if full_prune(command.catg) not in enabled and not admin:
                        raise PermissionError(f"This command is not enabled here. Use {prefix}ec to view or modify the list of enabled commands")
                    # argv is the raw parsed argument data
                    argv = comm[i:].strip()
                    run = True
                    print(f"{getattr(guild, 'id', 0)}: {user} ({u_id}) issued command {msg}")
                    req = command.min_level
                    fut = out_fut = None
                    try:
                        # Make sure server-only commands can only be run in servers.
                        if guild is None or getattr(guild, "ghost", None):
                            if getattr(command, "server_only", False):
                                raise ReferenceError("This command is only available in servers.")
                        # Make sure target has permission to use the target command, rate limit the command if necessary.
                        if not isnan(u_perm):
                            if not u_perm >= req:
                                raise command.perm_error(u_perm, req, "for command " + command_check)
                            x = command.rate_limit
                            if x:
                                x2 = x
                                if user.id in bot.owners:
                                    x = x2 = 0
                                elif isinstance(x, collections.abc.Sequence):
                                    x = x2 = x[not bot.is_trusted(getattr(guild, "id", 0))]
                                    x /= 2 ** bot.premium_level(user)
                                    x2 /= 2 ** bot.premium_level(user, absolute=True)
                                remaining += x
                                d = command.used
                                t = d.get(u_id, -inf)
                                wait = utc() - t - x
                                if wait > min(1 - x, -1):
                                    if x < x2 and (utc() - t - x2) < min(1 - x2, -1):
                                        bot.data.users.add_diamonds(user, (x - x2) / 100)
                                    if wait < 0:
                                        w = -wait
                                        d[u_id] = max(t, utc()) + w
                                        await asyncio.sleep(w)
                                    if len(d) >= 4096:
                                        with suppress(RuntimeError):
                                            d.pop(next(iter(d)))
                                    d[u_id] = max(t, utc())
                                else:
                                    raise TooManyRequests(f"Command has a rate limit of {sec2time(x)}; please wait {sec2time(-wait)}.")
                        flags = {}
                        if loop:
                            inc_dict(flags, h=1)
                        if argv:
                            # Commands by default always parse unicode fonts as regular text unless otherwise specified.
                            if not hasattr(command, "no_parse"):
                                argv = unicode_prune(argv)
                            argv = argv.strip()
                            # Parse command flags (this is a bit of a mess)
                            if hasattr(command, "flags"):
                                flaglist = command.flags
                                for q in "?-+":
                                    if q in argv:
                                        for char in flaglist:
                                            flag = q + char
                                            for r in (flag, flag.upper()):
                                                while len(argv) >= 4 and r in argv:
                                                    found = False
                                                    i = argv.index(r)
                                                    if i == 0 or argv[i - 1] == " " or argv[i - 2] == q:
                                                        with suppress(IndexError, KeyError):
                                                            if argv[i + 2] == " " or argv[i + 2] == q:
                                                                argv = argv[:i] + argv[i + 2:]
                                                                add_dict(flags, {char: 1})
                                                                found = True
                                                    if not found:
                                                        break
                                    if q in argv:
                                        for char in flaglist:
                                            flag = q + char
                                            for r in (flag, flag.upper()):
                                                while len(argv) >= 2 and r in argv:
                                                    found = False
                                                    for check in (r + " ", " " + r):
                                                        if check in argv:
                                                            argv = argv.replace(check, "")
                                                            add_dict(flags, {char: 1})
                                                            found = True
                                                    if argv == r:
                                                        argv = ""
                                                        add_dict(flags, {char: 1})
                                                        found = True
                                                    if not found:
                                                        break
                        if argv:
                            argv = argv.strip()
                        argl = None
                        # args is a list of arguments parsed from argv, using shlex syntax when possible.
                        if not argv:
                            args = []
                        else:
                            args = None
                            # Used as a goto lol
                            with suppress(StopIteration):
                                if not command or hasattr(command, "no_parse") or "argl" not in command.__call__.__code__.co_varnames:
                                    raise StopIteration
                                spl = argv.split(None, 2)
                                if len(spl) < 3:
                                    raise StopIteration
                                i = len(spl[0]) + len(spl[1]) + 2
                                brackets = {"<": ">", "(": ")", "[": "]", "{": "}"}
                                for x, y in brackets.items():
                                    if x in argv and y in argv:
                                        xi = argv.index(x)
                                        yi = argv.rindex(y)
                                        if xi < yi and yi >= i:
                                            checker = argv[xi:yi + 1]
                                            if regexp("<a?:[A-Za-z0-9\\-~_]+:[0-9]+>").search(checker) or regexp("<(?:@[!&]?|#)[0-9]+>").search(checker):
                                                continue
                                            middle = checker[1:-1]
                                            if len(middle.split(None, 1)) > 1 or "," in middle:
                                                if hasattr(command, "multi"):
                                                    argv2 = single_space((argv[:xi] + " " + argv[yi + 1:]).replace("\n", " ").replace(",", " ").replace("\t", " ")).strip()
                                                    argv3 = single_space(middle.replace("\n", " ").replace(",", " ").replace("\t", " ")).strip()
                                                    argl = smart_split(argv3)
                                                else:
                                                    argv2 = single_space(argv[:xi].replace("\n", " ").replace("\t", " ") + " " + (middle[xi + 1:yi]).replace("\n", " ").replace(",", " ").replace("\t", " ") + " " + argv[yi + 1:].replace("\n", " ").replace("\t", " "))
                                                args = smart_split(argv2)
                                                raise StopIteration
                            if args is None:
                                argv2 = single_space(argv.replace("\n", " ").replace("\t", " "))
                                args = smart_split(argv2)
                            if args and getattr(command, "flags", None):
                                if not ("a" in flags or "e" in flags or "d" in flags):
                                    if "a" in command.flags and "e" in command.flags and "d" in command.flags:
                                        if args[0].lower() in ("add", "enable", "set", "create", "append"):
                                            args.pop(0)
                                            argv = argv.split(None, 1)[-1]
                                            inc_dict(flags, a=1)
                                        elif args[0].lower() in ("rem", "disable", "remove", "unset", "delete"):
                                            args.pop(0)
                                            argv = argv.split(None, 1)[-1]
                                            inc_dict(flags, d=1)
                                    if args and "r" in command.flags:
                                        if args[0].lower() in ("clear", "reset"):
                                            args.pop(0)
                                            argv = argv.split(None, 1)[-1]
                                            inc_dict(flags, r=1)
                            args = list(args)
                        # Assign "guild" as an object that mimics the discord.py guild if there is none
                        if guild is None:
                            guild = self.UserGuild(
                                user=user,
                                channel=channel,
                            )
                            channel = guild.channel
                        # Automatically start typing if the command is time consuming
                        tc = getattr(command, "time_consuming", False)
                        if not loop and tc and not getattr(message, "simulated", False):
                            fut = create_task(discord.abc.Messageable.trigger_typing(channel))
                        # Get maximum time allowed for command to process
                        if isnan(u_perm):
                            timeout = None
                        else:
                            timeout = getattr(command, "_timeout_", 1) * bot.timeout
                            if timeout >= inf:
                                timeout = None
                            elif self.is_trusted(message.guild):
                                timeout *= 2
                            timeout *= 2 ** self.premium_level(user)
                        # Create a future to run the command
                        future = create_future(
                            command,                        # command is a callable object, may be async or not
                            bot=bot,                        # for interfacing with bot's database
                            argv=argv,                      # raw text argument
                            args=args,                      # split text arguments
                            argl=argl,                      # inputted array of arguments
                            flags=flags,                    # special flags
                            perm=u_perm,                    # permission level
                            user=user,                      # user that invoked the command
                            message=message,                # message data
                            channel=channel,                # channel data
                            guild=guild,                    # guild data
                            name=command_check,             # alias the command was called as
                            looped=loop,                    # whether this command was invoked as part of a loop
                            _timeout=timeout,               # timeout delay assigned to the command
                            timeout=timeout,                # timeout delay for the whole function
                        )
                        # Add a callback to typing in the channel if the command takes too long
                        if fut is None and not hasattr(command, "typing") and not getattr(message, "simulated", False):
                            create_task(delayed_callback(future, sqrt(3), discord.abc.Messageable.trigger_typing, channel, repeat=True))
                        if getattr(message, "slash", None):
                            create_task(delayed_callback(future, 1, self.defer_interaction, message))
                        with self.command_semaphore:
                            response = await future
                        # Send bot event: user has executed command
                        await self.send_event("_command_", user=user, command=command, loop=loop, message=message)
                        # Process response to command if there is one
                        if response is not None and not hasattr(response, "channel"):
                            if fut is not None:
                                await fut
                            # Raise exceptions returned by the command
                            if issubclass(type(response), Exception):
                                raise response
                            elif bool(response) is not False:
                                if callable(getattr(command, "_callback_", None)):
                                    if getattr(message, "slash", None):
                                        message.slash = False
                                # If 2-tuple returned, send as message-react pair
                                if type(response) is tuple and len(response) == 2:
                                    response, react = response
                                    if react == 1:
                                        react = "❎"
                                else:
                                    react = False
                                sent = None
                                # Process list as a sequence of messages to send
                                if type(response) is list or type(response) is alist and getattr(guild, "ghost", None):
                                    futs = deque()
                                    for r in response:
                                        async with Delay(1 / 3):
                                            futs.append(create_task(channel.send(r)))
                                    for fut in futs:
                                        await fut
                                elif type(response) is alist:
                                    m = guild.me
                                    futs = deque()
                                    for r in response:
                                        async with Delay(1 / 3):
                                            url = await self.get_proxy_url(m)
                                            futs.append(create_task(self.send_as_webhook(channel, r, username=m.display_name, avatar_url=url)))
                                    for fut in futs:
                                        await fut
                                # Process dict as kwargs for a message send
                                elif isinstance(response, collections.abc.Mapping):
                                    if "file" in response:
                                        sent = await self.send_with_file(channel, response.get("content", ""), **response, reference=message)
                                    else:
                                        sent = await send_with_react(channel, reference=not loop and message, **response)
                                else:
                                    if type(response) not in (str, bytes, bytearray):
                                        response = str(response)
                                    # Process everything else as a string
                                    if type(response) is str and (len(response) <= 2000 or getattr(message, "simulated", False)):
                                        sent = await send_with_react(channel, response, reference=not loop and message)
                                        # sent = await channel.send(response)
                                    else:
                                        # Send a file if the message is too long
                                        if type(response) is not bytes:
                                            response = bytes(str(response), "utf-8")
                                            filemsg = "Response too long for message."
                                        else:
                                            filemsg = "Response data:"
                                        f = CompatFile(response, filename="message.txt")
                                        sent = await self.send_with_file(channel, filemsg, f, reference=message)
                                # Add targeted react if there is one
                                if react and sent:
                                    await sent.add_reaction(react)
                    # Represents any timeout error that occurs
                    except (T0, T1, T2):
                        if fut is not None:
                            try:
                                await fut
                            except:
                                pass
                        print(msg)
                        raise TimeoutError("Request timed out.")
                    except (ArgumentError, TooManyRequests) as ex:
                        if fut is not None:
                            try:
                                await fut
                            except:
                                pass
                        command.used.pop(u_id, None)
                        out_fut = self.send_exception(channel, ex, message)
                        return remaining
                    # Represents all other errors
                    except Exception as ex:
                        if fut is not None:
                            try:
                                await fut
                            except:
                                pass
                        command.used.pop(u_id, None)
                        print_exc()
                        out_fut = self.send_exception(channel, ex, message, op=op)
                    if out_fut is not None and getattr(message, "simulated", None):
                        await out_fut
            elif getattr(message, "simulated", None):
                return -1
        # If message was not processed as a command, send a _nocommand_ event with the parsed message data.
        if not run:
            with self.command_semaphore:
                await self.send_event("_nocommand2_", message=message)
                not_self = True
                if u_id == bot.id:
                    not_self = False
                elif getattr(message, "webhook_id", None) and message.author.name == guild.me.display_name:
                    cola = await create_future(self.get_colour, self)
                    colb = await create_future(self.get_colour, message.author)
                    not_self = cola != colb
                if not_self:
                    temp = to_alphanumeric(cpy).casefold()
                    temp2 = to_alphanumeric(message.clean_content or message.content).casefold()
                    await self.send_event("_nocommand_", text=temp, text2=temp2, edit=edit, orig=orig, msg=msg, message=message, perm=u_perm, truemention=truemention)
        # Return the delay before the message can be called again. This is calculated by the rate limit of the command.
        return remaining

    @tracebacksuppressor
    async def process_http_command(self, t, name, nick, command):
        url = f"http://127.0.0.1:{PORT}/commands/{t}\x7f0"
        out = "[]"
        message = SimulatedMessage(self, command, t, name, nick)
        self.cache.users[message.author.id] = message.author
        after = await self.process_message(message, msg=command, slash=True)
        if after != -1:
            if after is not None:
                after += utc()
            else:
                after = 0
            for i in range(3600):
                if message.response:
                    break
                await asyncio.sleep(0.1)
            await self.react_callback(message, None, message.author)
            out = orjson.dumps(list(message.response))
        url = f"http://127.0.0.1:{PORT}/commands/{t}\x7f{after}"
        await Request(url, data=out, method="POST", headers={"Content-Type": "application/json"}, bypass=False, decode=True, aio=True, ssl=False)

    @tracebacksuppressor
    async def process_http_eval(self, t, proc):
        glob = self._globals
        url = f"http://127.0.0.1:{PORT}/commands/{t}\x7f0"
        out = '{"result":null}'
        code = None
        with suppress(SyntaxError):
            code = compile(proc, "<webserver>", "eval", optimize=2)
        if code is None:
            with suppress(SyntaxError):
                code = compile(proc, "<webserver>", "exec", optimize=2)
            if code is None:
                _ = glob.get("_")
                defs = False
                lines = proc.splitlines()
                for line in lines:
                    if line.startswith("def") or line.startswith("async def"):
                        defs = True
                func = "async def _():\n\tlocals().update(globals())\n"
                func += "\n".join(("\tglobals().update(locals())\n" if not defs and line.strip().startswith("return") else "") + "\t" + line for line in lines)
                func += "\n\tglobals().update(locals())"
                code2 = compile(func, "<webserver>", "exec", optimize=2)
                await create_future(eval, code2, glob)
                output = await glob["_"]()
                glob["_"] = _
        if code is not None:
            try:
                output = await create_future(eval, code, glob, priority=True)
            except:
                print(proc)
                raise
        if type(output) in (deque, alist):
            output = list(output)
        if output is not None:
            glob["_"] = output
        try:
            out = orjson.dumps(dict(result=output))
        except TypeError:
            try:
                out = json.dumps(dict(result=output), cls=MultiEncoder)
            except TypeError:
                out = orjson.dumps(dict(result=repr(output)))
        # print(url, out)
        await Request(url, data=out, method="POST", headers={"Content-Type": "application/json"}, bypass=False, decode=True, aio=True, ssl=False)

    async def load_guild_http(self, guild):
        _members = {}
        x = 0
        i = 1000
        while i >= 1000:
            for r in range(64):
                try:
                    async with self.load_semaphore:
                        memberdata = await Request(
                            f"https://discord.com/api/{api}/guilds/{guild.id}/members?limit=1000&after={x}",
                            authorise=True,
                            json=True,
                            aio=True,
                        )
                except:
                    print_exc()
                    await asyncio.sleep(r + 2)
                else:
                    break
            else:
                raise RuntimeError("Max retries exceeded in loading guild members via http.")
            members = {int(m["user"]["id"]): discord.Member(guild=guild, data=m, state=self._connection) for m in memberdata}
            _members.update(members)
            i = len(memberdata)
            x = max(members)
        guild._members = _members
        guild._member_count = len(_members)
        return guild.members

    @tracebacksuppressor
    async def load_guilds(self):
        funcs = [self._connection.chunk_guild, self.load_guild_http]
        futs = alist()
        for n, guild in enumerate(self.client.guilds):
            i = n % 5 != 0
            if not i and getattr(guild, "_member_count", len(guild._members)) > 250:
                i = 1
            fut = create_task(asyncio.wait_for(funcs[i](guild), timeout=None if i else 30))
            fut.guild = guild
            if len(futs) >= 8:
                pops = [a for a, fut in enumerate(futs) if fut.done()]
                futs.pops(pops)
                if len(futs) >= 8:
                    fut = futs.pop(0)
                    try:
                        await fut
                    except (T0, T1, T2):
                        print_exc()
                        await self.load_guild_http(fut.guild)
                    if "guilds" in self.data:
                        self.data.guilds.register(fut.guild)
            futs.append(fut)
        for fut in futs:
            try:
                await fut
            except (T0, T1, T2):
                print_exc()
                await self.load_guild_http(fut.guild)
            if "guilds" in self.data:
                self.data.guilds.register(fut.guild)
        self.users_updated = True
        print("Guilds loaded.")

    # Adds a webhook to the bot's user and webhook cache.
    def add_webhook(self, w):
        return self.data.webhooks.add(w)

    # Loads all webhooks in the target channel.
    def load_channel_webhooks(self, channel, force=False, bypass=False):
        return self.data.webhooks.get(channel, force=force, bypass=bypass)

    # Gets a valid webhook for the target channel, creating a new one when necessary.
    async def ensure_webhook(self, channel, force=False, bypass=False, fill=False):
        wlist = await self.load_channel_webhooks(channel, force=force, bypass=bypass)
        try:
            if fill:
                while len(wlist) < fill:
                    data = await self.get_request(get_author(self.user).url)
                    w = await channel.create_webhook(name=self.name, avatar=data, reason="Auto Webhook")
                    w = self.add_webhook(w)
                    wlist.append(w)
            if not wlist:
                data = await self.get_request(get_author(self.user).url)
                w = await channel.create_webhook(name=self.name, avatar=data, reason="Auto Webhook")
                w = self.add_webhook(w)
            else:
                w = wlist[0]
        except discord.HTTPException as ex:
            if "maximum" in str(ex).lower():
                print_exc()
                wlist = await self.load_channel_webhooks(channel, force=True, bypass=bypass)
                w = wlist.next()
        if not w.avatar or str(w.avatar) == "https://cdn.discordapp.com/embed/avatars/0.png":
            data = await self.get_request(get_author(self.user).url)
            return await w.edit(name=self.name, avatar=data)
        return w

    # Sends a message to the target channel, using a random webhook from that channel.
    async def send_as_webhook(self, channel, *args, recurse=True, **kwargs):
        if recurse and "exec" in self.data:
            try:
                avatar_url = kwargs.pop("avatar_url")
            except KeyError:
                pass
            else:
                with tracebacksuppressor:
                    kwargs["avatar_url"] = await self.data.exec.uproxy(avatar_url)
        if hasattr(channel, "simulated") or not getattr(channel, "guild", None) or hasattr(channel, "recipient") or not hasattr(channel, "send"):
            kwargs.pop("username", None)
            kwargs.pop("avatar_url", None)
            message = await discord.abc.Messageable.send(channel, *args, **kwargs)
            reacts = kwargs.pop("reacts", None)
        else:
            if args and args[0] and args[0].count(":") >= 2 and channel.guild.me.guild_permissions.manage_roles:
                everyone = channel.guild.default_role
                permissions = everyone.permissions
                if not permissions.use_external_emojis:
                    permissions.use_external_emojis = True
                    await everyone.edit(permissions=permissions, reason="I need to send emojis 🙃")
            mchannel = None
            for i in range(25):
                mchannel = channel.parent if hasattr(channel, "thread") or isinstance(channel, discord.Thread) else channel
                if not mchannel:
                    await asyncio.sleep(0.2)
                else:
                    break
            w = await self.ensure_webhook(mchannel, bypass=True)
            kwargs.pop("wait", None)
            reacts = kwargs.pop("reacts", None)
            try:
                async with getattr(w, "semaphore", emptyctx):
                    w = getattr(w, "webhook", w)
                    if hasattr(channel, "thread") or isinstance(channel, discord.Thread):
                        if kwargs.get("files"):
                            kwargs.pop("avatar_url", None)
                            kwargs.pop("username", None)
                            message = await channel.send(*args, **kwargs)
                        else:
                            data = dict(
                                content=args[0] if args else kwargs.get("content"),
                                username=kwargs.get("username"),
                                avatar_url=kwargs.get("avatar_url"),
                                tts=kwargs.get("tts"),
                                embeds=[emb.to_dict() for emb in kwargs.get("embeds", ())] or [kwargs["embed"].to_dict()] if kwargs.get("embed") is not None else None,
                            )
                            resp = await Request(
                                f"https://discord.com/api/{api}/webhooks/{w.id}/{w.token}?wait=True&thread_id={channel.id}",
                                method="POST",
                                authorise=True,
                                data=data,
                                aio=True,
                                json=True,
                            )
                            message = self.ExtendedMessage.new(resp)
                    else:
                        message = await w.send(*args, wait=True, **kwargs)
            except (discord.NotFound, discord.InvalidArgument, discord.Forbidden):
                w = await self.ensure_webhook(mchannel, force=True)
                async with getattr(w, "semaphore", emptyctx):
                    w = getattr(w, "webhook", w)
                    if hasattr(channel, "thread"):
                        resp = await Request(
                            f"https://discord.com/api/{api}/webhooks/{w.id}/{w.token}?wait=True&thread_id={channel.id}",
                            method="POST",
                            authorise=True,
                            data=data,
                            aio=True,
                            json=True,
                        )
                        message = self.ExtendedMessage.new(resp)
                    else:
                        message = await w.send(*args, wait=True, **kwargs)
            except discord.HTTPException as ex:
                if "400 Bad Request" in repr(ex):
                    if "embeds" in kwargs:
                        print(sum(len(e) for e in kwargs["embeds"]))
                        for embed in kwargs["embeds"]:
                            print(embed.to_dict())
                    print(args, kwargs)
                raise
            await self.seen(self.user, channel.guild, event="message", count=len(kwargs.get("embeds", (None,))), raw=f"Sending a message")
        if reacts:
            for react in reacts:
                await message.add_reaction(react)
        return message

    # Sends a list of embeds to the target sendable, using a webhook when possible.
    async def _send_embeds(self, sendable, embeds, reacts=None, reference=None, force=True, exc=True):
        s_id = verify_id(sendable)
        sendable = await self.fetch_messageable(s_id)
        if exc:
            ctx = self.ExceptionSender(sendable, reference=reference)
        else:
            ctx = tracebacksuppressor
        with ctx:
            if not embeds:
                return
            guild = getattr(sendable, "guild", None)
            # Determine whether to send embeds individually or as blocks of up to 10, based on whether it is possible to use webhooks
            if not guild:
                return await send_with_react(sendable, embeds=embeds, reacts=reacts, reference=reference)
            single = len(embeds) == 1
            if not hasattr(guild, "simulated") and hasattr(guild, "ghost"):
                single = True
            else:
                m = guild.me
                if not hasattr(guild, "simulated"):
                    if m is None:
                        m = self.user
                        single = True
                    else:
                        with suppress(AttributeError):
                            if not m.guild_permissions.manage_webhooks:
                                return await send_with_react(sendable, embeds=embeds, reacts=reacts, reference=reference)
            if single:
                for emb in embeds:
                    async with Delay(1 / 3):
                        if type(emb) is not discord.Embed:
                            emb = discord.Embed.from_dict(emb)
                            if not reacts:
                                reacts = emb.pop("reacts", None)
                        if reacts or reference:
                            create_task(send_with_react(sendable, embed=emb, reacts=reacts, reference=reference))
                        else:
                            create_task(send_with_reply(sendable, embed=emb))
                return
            if force:
                return await send_with_react(sendable, embeds=embeds, reacts=reacts, reference=reference)
            embs = deque()
            for emb in embeds:
                if type(emb) is not discord.Embed:
                    if not reacts:
                        reacts = emb.pop("reacts", None)
                    emb = discord.Embed.from_dict(emb)
                if len(embs) > 9 or len(emb) + sum(len(e) for e in embs) > 6000:
                    url = await self.get_proxy_url(m)
                    await self.send_as_webhook(sendable, embeds=embs, username=m.display_name, avatar_url=url, reacts=reacts)
                    embs.clear()
                embs.append(emb)
                reacts = None
            if embs:
                url = await self.get_proxy_url(m)
                await self.send_as_webhook(sendable, embeds=embs, username=m.display_name, avatar_url=url, reacts=reacts)

    async def defer_interaction(self, message):
        with suppress():
            if hasattr(message, "int_id"):
                int_id, int_token = message.int_id, message.int_token
            elif hasattr(message, "slash"):
                int_id, int_token = message.id, message.slash
            else:
                return
            await Request(
                f"https://discord.com/api/{api}/interactions/{int_id}/{int_token}/callback",
                method="POST",
                authorise=True,
                data='{"type":5}',
                aio=True,
            )
            message.deferred = True
            if self.cache.messages.get(message.id):
                self.cache.messages[message.id].deferred = True
            else:
                self.cache.messages[message.id] = message
            return message

    @tracebacksuppressor
    async def ignore_interaction(self, message):
        if hasattr(message, "int_id"):
            int_id, int_token = message.int_id, message.int_token
        elif hasattr(message, "slash"):
            int_id, int_token = message.id, message.slash
        else:
            return
        await Request(
            f"https://discord.com/api/{api}/interactions/{int_id}/{int_token}/callback",
            method="POST",
            authorise=True,
            data='{"type":6}',
            aio=True,
        )

    # Adds embeds to the embed sender, waiting for the next update event.
    def send_embeds(self, channel, embeds=None, embed=None, reacts=None, reference=None, exc=True):
        if embeds is not None and not issubclass(type(embeds), collections.abc.Collection):
            embeds = (embeds,)
        elif embeds:
            embeds = tuple(embeds)
        if embed is not None:
            if embeds:
                embeds += (embed,)
            else:
                embeds = (embed,)
        elif not embeds:
            return
        if reference:
            if getattr(reference, "slash", None):
                create_task(self.ignore_interaction(reference))
            if len(embeds) == 1:
                return create_task(send_with_react(channel, embed=embeds[0], reference=reference, reacts=reacts))
        c_id = verify_id(channel)
        user = self.cache.users.get(c_id)
        if user is not None:
            create_task(self._send_embeds(user, embeds, reacts, reference, exc=exc))
            return
        if reference:
            embeds = list(embeds)
            create_task(self._send_embeds(channel, [embeds.pop(0)], reacts, reference, exc=exc))
        if reacts:
            embeds = [e.to_dict() for e in embeds]
            for e in embeds:
                e["reacts"] = reacts
        embs = set_dict(self.embed_senders, c_id, [])
        embs.extend(embeds)
        if len(embs) > 2048:
            self.embed_senders[c_id] = embs[-2048:]

    def send_as_embeds(self, channel, description=None, title=None, fields=None, md=nofunc, author=None, footer=None, thumbnail=None, image=None, images=None, colour=None, reacts=None, reference=None, exc=True):
        if type(description) is discord.Embed:
            emb = description
            description = emb.description or None
            title = emb.title or None
            fields = emb.fields or None
            author = emb.author or None
            footer = emb.footer or None
            thumbnail = emb.thumbnail or None
            image = emb.image or None
            if emb.colour:
                colour = colorsys.rgb_to_hsv(*(alist(raw2colour(verify_id(emb.colour))) / 255))[0] * 1536
        if description and type(description) is not str:
            description = as_str(description)
        elif not description:
            description = None
        if not description and not fields and not thumbnail and not image and not images:
            return fut_nop
        return create_task(self._send_as_embeds(channel, description, title, fields, md, author, footer, thumbnail, image, images, colour, reacts, reference, exc=exc))

    async def _send_as_embeds(self, channel, description=None, title=None, fields=None, md=nofunc, author=None, footer=None, thumbnail=None, image=None, images=None, colour=None, reacts=None, reference=None, exc=True):
        fin_col = col = None
        if colour is None:
            if author:
                try:
                    url = author.icon_url
                except:
                    url = author.get("icon_url")
                if url:
                    with suppress():
                        fin_col = await self.data.colours.get(url)
        if fin_col is None:
            if type(colour) is discord.Colour:
                fin_col = colour
            elif colour is None:
                try:
                    colour = self.cache.colours[channel.id]
                except KeyError:
                    colour = xrand(12)
                self.cache.colours[channel.id] = (colour + 1) % 12
                fin_col = colour2raw(hue2colour(colour * 1536 / 12))
            else:
                col = colour if not issubclass(type(colour), collections.abc.Sequence) else colour[0]
                off = 128 if not issubclass(type(colour), collections.abc.Sequence) else colour[1]
                fin_col = colour2raw(hue2colour(col))
        embs = deque()
        emb = discord.Embed(colour=fin_col)
        if title:
            emb.title = title
        if author:
            try:
                emb.set_author(**author)
            except TypeError:
                emb.set_author(name=author.name, url=author.url, icon_url=author.icon_url)
        if description:
            # Separate text into paragraphs, then lines, then words, then characters and attempt to add them one at a time, adding extra embeds when necessary
            curr = ""
            if "\n\n" in description:
                paragraphs = alist(p + "\n\n" for p in description.split("\n\n"))
            else:
                paragraphs = alist((description,))
            while paragraphs:
                para = paragraphs.popleft()
                if len(para) > 4080:
                    temp = para[:4080]
                    try:
                        i = temp.rindex("\n")
                        s = "\n"
                    except ValueError:
                        try:
                            i = temp.rindex(" ")
                            s = " "
                        except ValueError:
                            paragraphs.appendleft(para[4080:])
                            paragraphs.appendleft(temp)
                            continue
                    paragraphs.appendleft(para[i + 1:])
                    paragraphs.appendleft(para[:i] + s)
                    continue
                if len(curr) + len(para) > 4080:
                    emb.description = md(curr.strip())
                    curr = para
                    embs.append(emb)
                    emb = discord.Embed(colour=fin_col)
                    if col is not None:
                        col += 128
                        emb.colour = colour2raw(hue2colour(col))
                else:
                    curr += para
            if curr:
                emb.description = md(curr.strip())
        if fields:
            if issubclass(type(fields), collections.abc.Mapping):
                fields = fields.items()
            for field in fields:
                if issubclass(type(field), collections.abc.Mapping):
                    field = tuple(field.values())
                elif not issubclass(type(field), collections.abc.Sequence):
                    try:
                        field = tuple(field)
                    except TypeError:
                        field = (field.name, field.value, getattr(field, "inline", None))
                n = lim_str(field[0], 256)
                v = lim_str(md(field[1]), 1024)
                i = True if len(field) < 3 else field[2]
                if len(emb) + len(n) + len(v) > 6000 or len(emb.fields) > 24:
                    embs.append(emb)
                    emb = discord.Embed(colour=fin_col)
                    if col is not None:
                        col += 128
                        emb.colour = colour2raw(hue2colour(col))
                    else:
                        try:
                            colour = self.cache.colours[channel.id]
                        except KeyError:
                            colour = xrand(12)
                        self.cache.colours[channel.id] = (colour + 1) % 12
                        emb.colour = colour2raw(hue2colour(colour * 1536 / 12))
                emb.add_field(name=n, value=v if v else "\u200b", inline=i)
        if len(emb):
            embs.append(emb)
        if thumbnail:
            embs[0].set_thumbnail(url=thumbnail)
        if footer and embs:
            embs[-1].set_footer(**footer)
        if image:
            if images:
                images = deque(images)
                images.appendleft(image)
            else:
                images = (image,)
        if images:
            for i, img in enumerate(images):
                if is_video(img):
                    create_task(channel.send(escape_roles(img)))
                else:
                    if i >= len(embs):
                        emb = discord.Embed(colour=fin_col)
                        if col is not None:
                            col += 128
                            emb.colour = colour2raw(hue2colour(col))
                        else:
                            try:
                                colour = self.cache.colours[channel.id]
                            except KeyError:
                                colour = xrand(12)
                            self.cache.colours[channel.id] = (colour + 1) % 12
                            emb.colour = colour2raw(hue2colour(colour * 1536 / 12))
                        embs.append(emb)
                    embs[i].set_image(url=img)
                    embs[i].url = img
        return self.send_embeds(channel, embeds=embs, reacts=reacts, reference=reference, exc=exc)

    # Updates all embed senders.
    def update_embeds(self, force=False):
        sent = False
        for s_id in self.embed_senders:
            embeds = self.embed_senders[s_id]
            if not force and len(embeds) <= 10 and sum(len(e) for e in embeds) <= 6000:
                continue
            embs = deque()
            for emb in embeds:
                if type(emb) is not discord.Embed:
                    emb = discord.Embed.from_dict(emb)
                # Send embeds in groups of up to 10, up to 6000 characters
                if len(embs) > 9 or len(emb) + sum(len(e) for e in embs) > 6000:
                    break
                embs.append(emb)
            # Left over embeds are placed back in embed sender
            self.embed_senders[s_id] = embeds = embeds[len(embs):]
            if not embeds:
                self.embed_senders.pop(s_id)
            create_task(self._send_embeds(s_id, embs, force=force, exc=False))
            sent = True
        return sent

    # The fast update loop that runs almost 24 times per second. Used for events where timing is important.
    async def fast_loop(self):

        async def event_call(freq):
            for i in range(freq):
                async with Delay(0.51 / freq):
                    await self.send_event("_call_")

        freq = 12
        sent = 0
        while not self.closed:
            with tracebacksuppressor:
                sent = self.update_embeds(utc() % 1 < 0.5)
                if sent:
                    await event_call(freq)
                else:
                    async with Delay(0.51 / freq):
                        await self.send_event("_call_")
                self.update_users()

    api_latency = inf
    # The slow update loop that runs once every 3 second2.
    async def slow_loop(self):
        await asyncio.sleep(2)
        errored = 0
        while not self.closed:
            async with Delay(3):
                async with tracebacksuppressor:
                    create_task(self.update_status())
                    with MemoryTimer("update_bytes"):
                        net = await create_future(psutil.net_io_counters)
                        net_bytes = net.bytes_sent + net.bytes_recv
                        if not hasattr(self, "net_bytes"):
                            self.net_bytes = deque(maxlen=3)
                            self.start_bytes = 0
                            if os.path.exists("saves/status.json"):
                                with tracebacksuppressor:
                                    with open("saves/status.json", "rb") as f:
                                        data = await create_future(f.read)
                                    status = eval(data)
                                    self.start_bytes = max(0, status["net_bytes"] - net_bytes)
                            if not self.start_bytes and os.path.exists("saves/status.json\x7f\x7f"):
                                with tracebacksuppressor:
                                    with open("saves/status.json\x7f\x7f", "rb") as f:
                                        data = await create_future(f.read)
                                    status = eval(data)
                                    self.start_bytes = max(0, status["net_bytes"] - net_bytes)
                        self.net_bytes.append(net_bytes)
                        self.bitrate = (self.net_bytes[-1] - self.net_bytes[0]) * 8 / len(self.net_bytes)
                        self.total_bytes = self.net_bytes[-1] + self.start_bytes
                    if xrand(2):
                        continue
                    try:
                        t = utc()
                        resp = await Request.sessions.next().head(f"https://discord.com/api/{api}")
                        self.activity += 1
                        self.api_latency = utc() - t
                    except:
                        if hasattr(self, "api_latency"):
                            self.api_latency *= 2
                        else:
                            self.api_latency = inf
                        if not errored:
                            print_exc()
                        errored = 2
                    else:
                        if errored > 0:
                            errored -= 1

    # The lazy update loop that runs once every 4-8 seconds.
    async def lazy_loop(self):
        await asyncio.sleep(5)
        while not self.closed:
            async with Delay(frand(4) + 4):
                async with tracebacksuppressor:
                    # self.var_count = await create_future(var_count)
                    with MemoryTimer("handle_update"):
                        await self.handle_update()

    # The slowest update loop that runs once every 5 minutes. Used for slow operations, such as the bot database autosave event.
    async def global_loop(self):
        while not self.closed:
            async with Delay(300):
                async with tracebacksuppressor:
                    await asyncio.sleep(1)
                    with MemoryTimer("update_file_cache"):
                        await create_future(update_file_cache)
                    await asyncio.sleep(1)
                    with MemoryTimer("get_disk"):
                        await self.get_disk()
                    with MemoryTimer("get_hosted"):
                        await self.get_hosted()
                    await asyncio.sleep(1)
                    with MemoryTimer("update_subs"):
                        await create_future(self.update_subs, priority=True)
                    await asyncio.sleep(1)
                    await self.send_event("_minute_loop_")
                    if SEMS:
                        for sem in tuple(SEMS.values()):
                            sem._update_bin()
                    create_future_ex(self.cache_reduce, priority=True)
                    await asyncio.sleep(1)
                    if self.server_init:
                        with tracebacksuppressor:
                            await Request(
                                f"http://127.0.0.1:{PORT}/api_update_replacers",
                                method="GET",
                                aio=True,
                                ssl=False,
                            )
                    with MemoryTimer("update"):
                        await create_future(self.update, priority=True)

    # Heartbeat loop: Repeatedly renames a file to inform the watchdog process that the bot's event loop is still running.
    @tracebacksuppressor
    async def heartbeat_loop(self):
        print("Heartbeat Loop initiated.")
        while not self.closed:
            async with Delay(0.2):
                d = os.path.exists(self.heartbeat)
                if d:
                    with tracebacksuppressor(FileNotFoundError, PermissionError):
                        os.rename(self.heartbeat, self.heartbeat_ack)

    # User seen event
    async def seen(self, *args, delay=0, event=None, **kwargs):
        for arg in args:
            if arg:
                await self.send_event("_seen_", user=arg, delay=delay, event=event, **kwargs)

    # Deletes own messages if any of the "X" emojis are reacted by a user with delete message permission level, or if the message originally contained the corresponding reaction from the bot.
    async def check_to_delete(self, message, reaction, user):
        if user.id == self.id:
            return
        if not message.reactions:
            message = await discord.abc.Messageable.fetch_message(message.channel, message.id)
            self.bot.add_message(message, files=False, force=True)
        if str(reaction) not in "❌✖️🇽❎🔳🔲":
            return
        if str(reaction) in "🔳🔲" and (not message.attachments and not message.embeds or "exec" not in self.data):
            return
        if message.author.id == self.id or getattr(message, "webhook_id", None):
            with suppress(discord.NotFound):
                u_perm = self.get_perms(user.id, message.guild)
                check = False
                if not u_perm < 3:
                    check = True
                elif message.reference and message.reference.resolved and message.reference.resolved.author.id == user.id:
                    check = True
                elif not message.reference:
                    for react in message.reactions:
                        if str(reaction) == str(react) and react.me:
                            check = True
                            break
                if check:
                    if str(reaction) in "🔳🔲":
                        if message.content.startswith("||"):
                            content = message.content.replace("||", "")
                        else:
                            urls = set()
                            for a in message.attachments:
                                url = await self.data.exec.uproxy(str(a.url))
                                urls.add(url)
                            for e in message.embeds:
                                if e.image:
                                    urls.add(best_url(e.image.url))
                                if e.thumbnail:
                                    urls.add(best_url(e.thumbnail))
                            symrem = "".maketrans({c: "" for c in "<>|*"})
                            spl = [word.translate(symrem) for word in message.content.split()]
                            content = " ".join(word for word in spl if not is_url(word))
                            if urls:
                                content += "\n" + "\n".join(f"||{url} ||" for url in urls)
                        before = copy.copy(message)
                        message = await message.edit(content=content, attachments=(), embeds=())
                        # await self.send_event("_edit_", before=before, after=message, force=True)
                    else:
                        await self.silent_delete(message, exc=True)
                        await self.send_event("_delete_", message=message)

    # Handles a new sent message, calls process_message and sends an error if an exception occurs.
    async def handle_message(self, message, edit=True):
        if message.author.id != self.user.id:
            for i, a in enumerate(message.attachments):
                if a.filename == "message.txt":
                    b = await self.get_request(message.attachments.pop(i).url)
                    if message.content:
                        message.content += " "
                    message.content += as_str(b)
        cpy = msg = message.content
        with self.ExceptionSender(message.channel, reference=message):
            if msg and msg[0] == "\\":
                cpy = msg[1:]
            await self.process_message(message, msg=cpy, edit=edit, orig=msg)

    def set_classes(self):
        bot = self

        # For compatibility with guild objects, takes a user and DM channel.
        class UserGuild(discord.Object):

            class UserChannel(discord.abc.PrivateChannel):

                def __init__(self, channel, **void):
                    self.channel = channel

                def __dir__(self):
                    data = set(object.__dir__(self))
                    data.update(dir(self.channel))
                    return data

                def __getattr__(self, key):
                    try:
                        return self.__getattribute__(key)
                    except AttributeError:
                        pass
                    return getattr(self.__getattribute__("channel"), key)

                def fetch_message(self, id):
                    return bot.fetch_message(id, self.channel)

                me = bot.user
                name = "DM"
                topic = None
                is_nsfw = lambda self: bot.is_nsfw(self.channel)
                is_news = lambda *self: False
                is_chanbel = True

            def __init__(self, user, channel, **void):
                self.channel = self.system_channel = self.rules_channel = self.UserChannel(channel)
                self.members = [user, bot.user]
                self._members = {m.id: m for m in self.members}
                self.channels = self.text_channels = [self.channel]
                self.voice_channels = []
                self.roles = []
                self.emojis = []
                self.get_channel = lambda *void1, **void2: self.channel
                self.owner_id = bot.id
                self.owner = bot.user
                self.fetch_member = bot.fetch_user
                self.get_member = lambda *void1, **void2: None
                self.voice_client = None

            def __dir__(self):
                data = set(object.__dir__(self))
                data.update(dir(self.channel))
                return data

            def __getattr__(self, key):
                try:
                    return self.__getattribute__(key)
                except AttributeError:
                    pass
                return getattr(self.__getattribute__("channel"), key)
            
            get_role = lambda *args: None

            filesize_limit = 8388608
            bitrate_limit = 98304
            emoji_limit = 0
            large = False
            description = ""
            max_members = 2
            unavailable = False
            ghost = True
            is_channel = True

        # Represents a deleted/not found user.
        class GhostUser(discord.abc.Snowflake):

            __repr__ = lambda self: f"<Ghost User id={self.id} name='{self.name}' discriminator='{self.discriminator}' bot=False>"
            __str__ = lambda self: f"{self.name}#{self.discriminator}"
            system = False
            history = lambda *void1, **void2: fut_nop
            dm_channel = None
            create_dm = lambda self: fut_nop
            relationship = None
            is_friend = lambda self: None
            is_blocked = lambda self: None
            colour = color = discord.Colour(16777215)
            avatar = _avatar = ""
            name = "[USER DATA NOT FOUND]"
            nick = None
            discriminator = "0000"
            id = 0
            guild = None
            status = None
            voice = None
            display_avatar = avatar_url = icon_url = url = bot.discord_icon
            joined_at = premium_since = None
            _client_status = {None: "offline"}
            pending = False
            ghost = True
            roles = ()
            _roles = ()
            activities = ()
            _activities = ()
            public_flags = discord.flags.PublicUserFlags()
            _public_flags = discord.flags.PublicUserFlags()
            banner = None
            _banner = None
            accent_colour = None
            _accent_colour = None

            def __getattr__(self, k):
                if k == "member":
                    return self.__getattribute__(k)
                elif hasattr(self, "member"):
                    try:
                        return getattr(self.member, k)
                    except AttributeError:
                        pass
                return self.__getattribute__(k)

            @property
            def display_name(self):
                return self.nick or self.name

            @property
            def mention(self):
                return f"<@{self.id}>"

            @property
            def created_at(self):
                return snowflake_time_3(self.id)

            @property
            def _state(self):
                return bot._state

            @property
            def _user(self):
                return bot.cache.users.get(self.id) or self
            @_user.setter
            def set_user(self, user):
                bot.cache.users[user.id] = user

            def _to_minimal_user_json(self):
                return cdict(
                    username=self.name,
                    id=self.id,
                    avatar=self.avatar,
                    discriminator=self.discriminator,
                    bot=self.bot,
                )

            def _update(self, data):
                m = discord.Member._copy(self)
                m._update(data)
                self.member = m
                return m

        GhostUser.bot = False

        # Represents a deleted/not found message.
        class GhostMessage(discord.abc.Snowflake):

            content = bold(css_md(uni_str("[MESSAGE DATA NOT FOUND]"), force=True))

            def __init__(self):
                self.author = bot.get_user(bot.deleted_user)
                self.channel = None
                self.guild = None
                self.id = 0

            async def delete(self, *void1, **void2):
                pass

            __repr__ = lambda self: self.system_content or self.content
            tts = False
            type = "default"
            nonce = False
            embeds = ()
            call = None
            mention_everyone = False
            mentions = ()
            webhook_id = None
            attachments = ()
            pinned = False
            flags = None
            reactions = ()
            activity = None
            system_content = clean_content = ""
            edited_at = None
            jump_url = "https://discord.com/channels/-1/-1/-1"
            is_system = lambda self: None

            @property
            def created_at(self):
                return snowflake_time_3(self.id)

            edit = delete
            publish = delete
            pin = delete
            unpin = delete
            add_reaction = delete
            remove_reaction = delete
            clear_reaction = delete
            clear_reactions = delete
            ack = delete
            ghost = True
            deleted = True

        class ExtendedMessage:

            __slots__ = ("__dict__",)

            @classmethod
            def new(cls, data, channel=None, **void):
                if not channel:
                    try:
                        channel = bot.force_channel(data)
                    except:
                        print_exc()
                message = discord.Message(channel=channel, data=copy.deepcopy(data), state=bot._state)
                apply_stickers(message, data)
                self = cls(message)
                self._data = data
                return self

            async def edit(self, *args, **kwargs):
                if not self.webhook_id:
                    return await discord.Message.edit(self, *args, **kwargs)
                try:
                    w = bot.cache.users[self.webhook_id]
                    webhook = getattr(w, "webhook", w)
                except KeyError:
                    webhook = await bot.fetch_webhook(self.webhook_id)
                    bot.data.webhooks.add(webhook)
                if webhook.id == bot.id:
                    return await discord.Message.edit(self, *args, **kwargs)
                data = kwargs
                if args:
                    data["content"] = " ".join(args)
                if "embed" in data:
                    data["embeds"] = [data.pop("embed").to_dict()]
                elif "embeds" in data:
                    data["embeds"] = [emb.to_dict() for emb in data["embeds"]]
                resp = await Request(
                    f"https://discord.com/api/{api}/webhooks/{webhook.id}/{webhook.token}/messages/{self.id}",
                    data=data,
                    authorise=True,
                    method="PATCH",
                    aio=True,
                    json=True,
                )
                message = self.__class__.new(channel=self.channel, data=resp)
                return bot.add_message(message, force=2)

            def __init__(self, message):
                self.message = message
            
            def __copy__(self):
                return self.__class__(copy.copy(self.message))

            def __getattr__(self, k):
                if k == "channel":
                    m = object.__getattribute__(self, "message")
                    c = getattr(m, "channel")
                    if isinstance(c, discord.abc.PrivateChannel):
                        try:
                            c = bot.cache.channels[c.id]
                        except KeyError:
                            pass
                        else:
                            m.channel = c
                    return c
                elif k == "guild":
                    m = object.__getattribute__(self, "message")
                    g = getattr(m, "guild")
                    if not g:
                        c = self.channel
                        m.guild = getattr(c, "guild", None)
                    return m.guild
                try:
                    return self.__getattribute__(k)
                except AttributeError:
                    pass
                m = object.__getattribute__(self, "message")
                if not isinstance(m, self.__class__):
                    return getattr(m, k)
                return object.__getattribute__(m, k)

        class LoadedMessage(discord.Message):

            def __getattr__(self, k):
                if k not in ("mentions", "role_mentions"):
                    return super().__getattribute__(k)
                try:
                    return super().__getattribute__(k)
                except AttributeError:
                    return []

        class CachedMessage(discord.abc.Snowflake):

            __slots__ = ("_data", "id", "created_at", "author", "channel", "channel_id", "deleted", "attachments", "sem")

            def __init__(self, data):
                self._data = data
                self.id = int(data["id"])
                self.created_at = snowflake_time_3(self.id)
                author = data["author"]
                if author["id"] not in bot.cache.users:
                    bot.user2cache(author)

            def __copy__(self):
                d = dict(self.__getattribute__("_data"))
                channel = self.channel
                if "channel_id" not in d:
                    d["channel_id"] = channel.id
                author = self.author
                if "tts" not in d:
                    d["tts"] = False
                if "message_reference" in d:
                    with tracebacksuppressor:
                        ref = d["message_reference"]
                        if "channel_id" not in ref:
                            ref["channel_id"] = d["channel_id"]
                        if "guild_id" not in ref:
                            if hasattr(channel, "guild"):
                                ref["guild_id"] = channel.guild.id
                        if d.get("referenced_message"):
                            m = d["referenced_message"]
                            m["channel_id"] = d["channel_id"]
                            if "message_reference" in m:
                                m["message_reference"]["channel_id"] = d["channel_id"]
                for k in ("tts", "pinned", "mention_everyone"):
                    if k not in d:
                        d[k] = None
                if "edited_timestamp" not in d:
                    d["edited_timestamp"] = ""
                if "type" not in d:
                    d["type"] = 0
                if "content" not in d:
                    d["content"] = ""
                for k in ("reactions", "attachments", "embeds"):
                    if k not in d:
                        d[k] = []
                try:
                    message = bot.LoadedMessage(state=bot._state, channel=channel, data=d)
                except:
                    print(d)
                    raise
                apply_stickers(message, d)
                if not getattr(message, "author", None):
                    message.author = author
                return message

            def __getattr__(self, k):
                if k in self.__slots__:
                    try:
                        return self.__getattribute__(k)
                    except AttributeError:
                        if k == "deleted":
                            raise
                if k in ("simulated", "slash"):
                    raise AttributeError(k)
                d = self.__getattribute__("_data")
                if k in ("content", "system_content", "clean_content"):
                    return d.get(k) or d.get("content", "")
                if k == "channel":
                    try:
                        channel, _ = bot._get_guild_channel(d)
                        if channel is None:
                            raise LookupError
                    except LookupError:
                        pass
                    else:
                        return channel
                    cid = int(d["channel_id"])
                    channel = bot.cache.channels.get(cid)
                    if channel:
                        self.channel = channel
                        return channel
                    else:
                        return cdict(id=cid)
                if k == "guild":
                    return getattr(self.channel, "guild", None)
                if k == "author":
                    self.author = bot.get_user(d["author"]["id"], replace=True)
                    guild = getattr(self.channel, "guild", None)
                    if guild is not None:
                        member = guild.get_member(self.author.id)
                        if member is not None:
                            self.author = member
                    return self.author
                if k == "type":
                    return discord.enums.try_enum(discord.MessageType, d.get("type", 0))
                if k == "attachments":
                    self.attachments = [discord.Attachment(data=a, state=bot._state) for a in d.get("attachments", ())]
                    apply_stickers(self, d)
                if k == "embeds":
                    return [discord.Embed.from_dict(a) for a in d.get("embeds", ())]
                if k == "system_content" and not d.get("type"):
                    return self.content
                m = bot.cache.messages.get(d["id"])
                if m is None or m is self or not isinstance(m, bot.LoadedMessage):
                    message = self.__copy__()
                    if type(m) not in (discord.Message, bot.ExtendedMessage):
                        bot.add_message(message, files=False)
                    try:
                        return getattr(message, k)
                    except (AttributeError, KeyError):
                        if k == "mentions":
                            return ()
                        raise
                try:
                    return getattr(m, k)
                except AttributeError:
                    if k == "mentions":
                        return ()
                    raise

        class MessageCache(collections.abc.Mapping):
            data = bot.cache.messages

            def __getitem__(self, k):
                with suppress(KeyError):
                    return self.data[k]
                if "message_cache" in bot.data:
                    return bot.data.message_cache.load_message(k)
                raise KeyError(k)
            __call__ = __getitem__

            def __setitem__(self, k, v):
                bot.add_message(v, force=True)

            __delitem__ = data.pop
            __bool__ = lambda self: bool(self.data)
            __iter__ = data.__iter__
            __reversed__ = data.__reversed__
            __len__ = data.__len__
            __str__ = lambda self: f"<MessageCache ({len(self)} items)>"
            __repr__ = data.__repr__

            def __contains__(self, k):
                with suppress(KeyError):
                    self[k]
                    return True
                return False

            get = data.get
            pop = data.pop
            popitem = data.popitem
            clear = data.clear

        # A context manager that sends exception tracebacks to a sendable.
        class ExceptionSender(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

            def __init__(self, sendable, *args, reference=None, **kwargs):
                self.sendable = sendable
                self.reference = reference
                self.exceptions = args + tuple(kwargs.values())

            def __exit__(self, exc_type, exc_value, exc_tb):
                if exc_type and exc_value:
                    for exception in self.exceptions:
                        if issubclass(type(exc_value), exception):
                            bot.send_exception(self.sendable, exc_value, self.reference)
                            return True
                    bot.send_exception(self.sendable, exc_value, self.reference)
                    with tracebacksuppressor:
                        raise exc_value
                return True
            
            __aexit__ = lambda self, *args: as_fut(self.__exit__(*args))
            __call__ = lambda self, *args, **kwargs: self.__class__(*args, **kwargs)


        bot.UserGuild = UserGuild
        bot.GhostUser = GhostUser
        bot.GhostMessage = GhostMessage
        bot.ExtendedMessage = ExtendedMessage
        bot.LoadedMessage = LoadedMessage
        bot.CachedMessage = CachedMessage
        bot.MessageCache = MessageCache
        bot.ExceptionSender = ExceptionSender

    def monkey_patch(self):
        bot = self

        discord.http.Route.BASE = f"https://discord.com/api/{api}"
        discord.Member.permissions_in = lambda self, channel: discord.Permissions.none() if not getattr(channel, "permissions_for", None) else channel.permissions_for(self)
        discord.VoiceChannel._get_channel = lambda self: as_fut(self)

        async def received_message(self, msg, /):
            if type(msg) is bytes:
                self._buffer.extend(msg)
                if len(msg) < 4 or msg[-4:] != b"\x00\x00\xff\xff":
                    return
                msg = self._zlib.decompress(self._buffer)
                self._buffer = bytearray()
            msg = orjson.loads(msg)
            bot.socket_responses.append(msg)
            self._dispatch("socket_response", msg)
            event = msg.get("t")
            op = msg.get("op")
            data = msg.get("d")
            seq = msg.get("s")
            if seq is not None:
                self.sequence = seq
            if self._keep_alive:
                self._keep_alive.tick()
            if op != self.DISPATCH:
                if op == self.RECONNECT:
                    await self.close()
                    raise discord.gateway.ReconnectWebSocket(self.shard_id)
                if op == self.HEARTBEAT_ACK:
                    if self._keep_alive:
                        self._keep_alive.ack()
                    return
                if op == self.HEARTBEAT:
                    if self._keep_alive:
                        beat = self._keep_alive.get_payload()
                        await self.send_as_json(beat)
                    return
                if op == self.HELLO:
                    interval = data["heartbeat_interval"] / 1000
                    self._keep_alive = discord.gateway.KeepAliveHandler(ws=self, interval=interval, shard_id=self.shard_id)
                    await self.send_as_json(self._keep_alive.get_payload())
                    return self._keep_alive.start()
                if op == self.INVALIDATE_SESSION:
                    if data is True:
                        await self.close()
                        raise discord.gateway.ReconnectWebSocket(self.shard_id)
                    self.sequence = None
                    self.session_id = None
                    print(f"Shard ID {self.shard_id} session has been invalidated.")
                    await self.close(code=1000)
                    raise discord.gateway.ReconnectWebSocket(self.shard_id, resume=False)
                return print(f"Unknown OP code {op}.")
            if event == "READY":
                self._trace = trace = data.get("_trace", [])
                self.session_id = data["session_id"]
                data["__shard_id__"] = self.shard_id
            elif event == "RESUMED":
                self._trace = trace = data.get("_trace", [])
                data["__shard_id__"] = self.shard_id
            try:
                func = self._discord_parsers[event]
            except KeyError:
                print(f"Unknown event {event}.", data)
                self._discord_parsers[event] = lambda data: None
            else:
                func(data)
            removed = deque()
            for index, entry in enumerate(self._dispatch_listeners):
                if entry.event != event:
                    continue
                future = entry.future
                if future.cancelled():
                    removed.append(index)
                    continue
                try:
                    valid = entry.predicate(data)
                except Exception as exc:
                    future.set_exception(exc)
                    removed.append(index)
                else:
                    if valid:
                        ret = data if entry.result is None else entry.result(data)
                        future.set_result(ret)
                        removed.append(index)
            for index in reversed(removed):
                del self._dispatch_listeners[index]

        discord.gateway.DiscordWebSocket.received_message = lambda self, msg: received_message(self, msg)

        def _get_guild_channel(self, data):
            channel_id = int(data["channel_id"])
            try:
                channel = bot.cache.channels[channel_id]
            except KeyError:
                pass
            else:
                return channel, getattr(channel, "guild", None)
            try:
                guild = self._get_guild(int(data["guild_id"]))
            except KeyError:
                channel = discord.DMChannel._from_message(self, channel_id)
                guild = None
            else:
                channel = guild and guild._resolve_channel(channel_id)
            return channel or discord.PartialMessageable(state=self, id=channel_id), guild

        discord.state.ConnectionState._get_guild_channel = lambda self, data: _get_guild_channel(self, data)

        async def get_gateway(self, *, encoding="json", zlib=True):
            try:
                data = await self.request(discord.http.Route("GET", "/gateway"))
            except discord.HTTPException as exc:
                raise discord.GatewayNotFound() from exc
            value = "{0}?encoding={1}&v=" + api[1:]
            if zlib:
                value += "&compress=zlib-stream"
            return value.format(data["url"], encoding)

        discord.http.HTTPClient.get_gateway = lambda self, *args, **kwargs: get_gateway(self, *args, **kwargs)

        async def fill_messages(self):
            if not getattr(self, "channel", None):
                self.channel = await self.messageable._get_channel()
            if self._get_retrieve():
                try:
                    data = await self._retrieve_messages(self.retrieve)
                except TypeError:
                    data = await self._retrieve_messages(self.retrieve)
                if len(data) < 100:
                    self.limit = 0
                if self.reverse:
                    data = reversed(data)
                if self._filter:
                    data = filter(self._filter, data)
                c_id = self.channel.id
                CM = bot.CachedMessage
                for element in data:
                    element["channel_id"] = c_id
                    message = CM(element)
                    message.channel = self.channel
                    await self.messages.put(message)
        
        discord.iterators.HistoryIterator.fill_messages = lambda self: fill_messages(self)

        def parse_message_reaction_add(self, data):
            emoji = data["emoji"]
            emoji_id = utils._get_as_snowflake(emoji, "id")
            emoji = discord.PartialEmoji.with_state(self, id=emoji_id, animated=emoji.get("animated", False), name=emoji["name"])
            raw = discord.RawReactionActionEvent(data, emoji, "REACTION_ADD")

            member_data = data.get("member")
            if member_data:
                guild = self._get_guild(raw.guild_id)
                raw.member = discord.Member(data=member_data, guild=guild, state=self)
            else:
                raw.member = None
            self.dispatch("raw_reaction_add", raw)
            create_task(bot.reaction_add(raw, data))

        discord.state.ConnectionState.parse_message_reaction_add = lambda self, data: parse_message_reaction_add(self, data)

        def parse_message_reaction_remove(self, data):
            emoji = data["emoji"]
            emoji_id = utils._get_as_snowflake(emoji, "id")
            emoji = discord.PartialEmoji.with_state(self, id=emoji_id, animated=emoji.get("animated", False), name=emoji["name"])
            raw = discord.RawReactionActionEvent(data, emoji, "REACTION_REMOVE")
            self.dispatch("raw_reaction_remove", raw)
            create_task(bot.reaction_remove(raw, data))

        discord.state.ConnectionState.parse_message_reaction_remove = lambda self, data: parse_message_reaction_remove(self, data)

        def parse_message_reaction_remove_all(self, data):
            raw = discord.RawReactionClearEvent(data)
            self.dispatch("raw_reaction_clear", raw)
            create_task(bot.reaction_clear(raw, data))
        
        discord.state.ConnectionState.parse_message_reaction_remove_all = lambda self, data: parse_message_reaction_remove_all(self, data)

        def parse_message_create(self, data):
            message = bot.ExtendedMessage.new(data)
            self.dispatch("message", message)
            if self._messages is not None:
                self._messages.append(message)
            channel = message.channel
            if channel:
                try:
                    if not getattr(channel, "guild", None):
                        channel.guild = message.guild
                    channel.last_message_id = message.id
                except AttributeError:
                    pass

        discord.state.ConnectionState.parse_message_create = lambda self, data: parse_message_create(self, data)

        def create_message(self, channel, data):
            data["channel_id"] = channel.id
            return bot.ExtendedMessage.new(data)
        
        discord.state.ConnectionState.create_message = lambda self, *, channel, data: create_message(self, channel, data)

        @property
        def cached_message(self):
            m_id = self.message_id
            m = bot.cache.messages.get(m_id)
            if m:
                return m
            return bot.data.message_cache.load_message(m_id)

        discord.message.MessageReference.cached_message = cached_message

        async def _request(self, bucket, files, form, method, url, kwargs, lock, maybe_lock=None):
            for tries in range(5):
                if files:
                    for f in files:
                        f.reset(seek=tries)
                if form:
                    form_data = aiohttp.FormData()
                    for params in form:
                        form_data.add_field(**params)
                    kwargs['data'] = form_data

                try:
                    bot.activity += 1
                    r = await Request.sessions.next().request(method.upper(), url, **kwargs)
                    data = await discord.http.json_or_text(r)
                    status = r.status

                    if maybe_lock and status != 429:
                        # check if we have rate limit header information
                        remaining = r.headers.get('X-Ratelimit-Remaining')
                        if remaining == '0':
                            # we've depleted our current bucket
                            delta = parse_ratelimit_header(r.headers)
                            # log.debug('A rate limit bucket has been exhausted (bucket: %s, retry: %s).', bucket, delta)
                            maybe_lock.defer()
                            self.loop.call_later(delta, lock.release)
                        if remaining and method.lower() != "patch":
                            limit = r.headers.get('X-Ratelimit-Limit')
                            if limit and int(limit) == int(remaining) + 1:
                                retry_after = r.headers.get("Retry-After") or r.headers.get("X-Ratelimit-Reset-After")
                                if retry_after and float(retry_after):
                                    limit = int(limit)
                                    retry_after = round_min(retry_after) + limit * 0.03
                                    sem = Semaphore(
                                        limit,
                                        256,
                                        rate_limit=retry_after,
                                        weak=True,
                                    )
                                    async with sem:
                                        self._locks[bucket] = sem
                                    maybe_lock = None
                                    # print("Successfully registered hard rate limit", sem, "for", bucket)

                    # the request was successful so just return the text/json
                    if status in range(200, 400):
                        return data

                    # we are being rate limited
                    if status == 429:
                        if not r.headers.get('Via'):
                            # Banned by Cloudflare more than likely.
                            raise discord.HTTPException(r, data)
                        # fmt = 'We are being rate limited. Retrying in %.2f seconds. Handled under the bucket "%s"'
                        delta = parse_ratelimit_header(r.headers)
                        is_global = data.get('global', False)
                        if is_global:
                            # log.warning('Global rate limit has been hit. Retrying in %.2f seconds.', retry_after)
                            self._global_over.clear()
                        if not maybe_lock:
                            lock.delay_for(delta)
                        await asyncio.sleep(delta)
                        if not maybe_lock:
                            fut = create_task(lock.__aenter__())
                            lock.__exit__()
                            await fut
                        if is_global:
                            self._global_over.set()
                            # log.debug('Global rate limit is now over.')
                        if tries < 3:
                            continue
                        print(url)
                        raise ConnectionError(429, r, data)

                    # we've received a 500 or 502, unconditional retry
                    if status >= 500 and tries < 3:
                        await asyncio.sleep(1 + tries * 2)
                        continue

                    if not hasattr(r, "reason"):
                        r.reason = as_str(data)
                    # the usual error cases
                    if status == 403:
                        raise discord.Forbidden(r, data)
                    elif status == 404:
                        raise discord.NotFound(r, data)
                    elif status == 503:
                        raise discord.DiscordServerError(r, data)
                    else:
                        raise discord.HTTPException(r, data)

                # This is handling exceptions from the request
                except (OSError, discord.DiscordServerError):
                    # Connection reset by peer
                    if tries < 4:
                        await asyncio.sleep(1 + tries * 2)
                        continue
                    if utc() - Request.ts > 720:
                        print("Reinitialising request client...")
                        await Request._init_()
                    raise
            raise RuntimeError("Somehow ran out of attempts then used one more")
    
        async def request(self, route, *, files=None, form=None, **kwargs):
            bucket = route.bucket
            method = route.method
            url = route.url

            lock = self._locks.get(bucket)
            if lock is None:
                if "/reactions" in route.path:
                    lock = Semaphore(1, 16, rate_limit=0.28)
                elif "/messages" in route.path:
                    if method.lower() == "post":
                        lock = Semaphore(5, 256, rate_limit=5.15)
                    elif method.lower() == "patch":
                        m_id = verify_id(url.rsplit("/", 1)[-1])
                        try:
                            message = await bot.fetch_message(m_id)
                        except LookupError:
                            pass
                        else:
                            lock = getattr(message, "sem", None)
                            if not lock and message.edited_at and (utc_ddt() - message.created_at).total_seconds() >= 3590:
                                lock = message.sem = Semaphore(3, 1, rate_limit=20.09)
                    # elif method.lower() == "patch":
                    #     m_id = int(route.url.rsplit("/", 1)[-1])
                    #     td = datetime.datetime.now() - snowflake_time_2(m_id)
                    #     if td.total_seconds() > 3599:
                    #         lock = Semaphore(5, 256, rate_limit=5.15)
                if not lock:
                    lock = asyncio.Lock()
                self._locks[bucket] = lock

            # header creation
            headers = {'User-Agent': self.user_agent}
            if self.token is not None:
                headers['Authorization'] = 'Bot ' + self.token
            # some checking if it's a JSON request
            if 'json' in kwargs:
                headers['Content-Type'] = 'application/json'
                kwargs['data'] = utils._to_json(kwargs.pop('json'))

            try:
                reason = kwargs.pop('reason')
            except KeyError:
                pass
            else:
                if reason:
                    headers['X-Audit-Log-Reason'] = urllib.parse.quote(reason, safe='/ ')
            kwargs['headers'] = headers

            # Proxy support
            if self.proxy is not None:
                kwargs['proxy'] = self.proxy
            if self.proxy_auth is not None:
                kwargs['proxy_auth'] = self.proxy_auth

            if not self._global_over.is_set():
                # wait until the global lock is complete
                await self._global_over.wait()

            if not isinstance(lock, Semaphore):
                await lock.acquire()
                lock = self._locks.get(bucket)
                if not isinstance(lock, Semaphore):
                    with discord.http.MaybeUnlock(lock) as maybe_lock:
                        return await _request(self, bucket, files, form, method, url, kwargs, lock, maybe_lock=maybe_lock)
            if isinstance(lock, Semaphore):
                async with lock:
                    return await _request(self, bucket, files, form, method, url, kwargs, lock)

            # We've run out of retries, raise.
            if r.status >= 500:
                raise discord.DiscordServerError(r, data)
            raise discord.HTTPException(r, data)

        discord.http.HTTPClient.request = lambda self, *args, **kwargs: request(self, *args, **kwargs)

    def send_exception(self, messageable, ex, reference=None, op=None):
        if getattr(ex, "no_react", None):
            reacts = ""
        else:
            reacts="❎"
        footer = None
        fields = None
        if isinstance(ex, TooManyRequests):# and not random.randint(0, 5):
            fields = (("Running into the rate limit often?", f"Consider donating using one of the subscriptions from my [ko-fi]({self.kofi_url}), which will grant shorter rate limits amongst many feature improvements!"),)
        elif isinstance(op, tuple):
            fields = (op,)
        else:
            fields = (("Unexpected or confusing error?", f"Use {self.get_prefix(getattr(messageable, 'guild', None))}help for help, or consider joining the [support server]({self.rcc_invite}) for bug reports!"),)
        if reference and isinstance(ex, discord.Forbidden) and reference.guild and not messageable.permissions_for(reference.guild.me).send_messages:
            return create_task(self.missing_perms(messageable, reference))
        print(reference)
        return self.send_as_embeds(
            messageable,
            description="\n".join(as_str(i) for i in ex.args),
            title=f"⚠ {type(ex).__name__} ⚠",
            fields=fields,
            reacts=reacts,
            reference=reference,
            footer=footer,
            exc=False,
        )

    async def missing_perms(self, messageable, reference):
        guild = messageable.guild
        channel = messageable
        message = reference
        user = message.author
        s = f"Oops, it appears I do not have permission to reply to your command [here](https://discord.com/channels/{guild.id}/{channel.id}/{message.id}).\nPlease contact an admin of the server if you believe this is a mistake!"
        colour = await self.get_colour(self.user)
        emb = discord.Embed(colour=colour)
        emb.set_author(**get_author(self.user))
        emb.description = s
        return await user.send(embed=emb)

    async def reaction_add(self, raw, data):
        channel = await self.fetch_channel(raw.channel_id)
        user = await self.fetch_user(raw.user_id)
        emoji = self._upgrade_partial_emoji(raw.emoji)
        try:
            message = await self.fetch_message(raw.message_id)
        except LookupError:
            message = await discord.abc.Messageable.fetch_message(channel, raw.message_id)
            reaction = message._add_reaction(data, emoji, user.id)
            if reaction.count > 1:
                reaction.count -= 1
        else:
            reaction = message._add_reaction(data, emoji, user.id)
        self.dispatch("reaction_add", reaction, user)
        self.add_message(message, files=False, force=True)

    async def reaction_remove(self, raw, data):
        channel = await self.fetch_channel(raw.channel_id)
        user = await self.fetch_user(raw.user_id)
        emoji = self._upgrade_partial_emoji(raw.emoji)
        reaction = None
        try:
            message = await self.fetch_message(raw.message_id)
        except LookupError:
            message = await discord.abc.Messageable.fetch_message(channel, raw.message_id)
            reaction = message._add_reaction(data, emoji, user.id)
            if reaction.count > 1:
                reaction.count -= 1
        else:
            with tracebacksuppressor(ValueError):
                reaction = message._remove_reaction(data, emoji, user.id)
        if not reaction:
            message = await discord.abc.Messageable.fetch_message(channel, raw.message_id)
            reaction = message._add_reaction(data, emoji, user.id)
            if reaction.count > 1:
                reaction.count -= 1
            reaction = message._remove_reaction(data, emoji, user.id)
        self.dispatch("reaction_remove", reaction, user)
        self.add_message(message, files=False, force=True)
    
    async def reaction_clear(self, raw, data):
        channel = await self.fetch_channel(raw.channel_id)
        message = await self.fetch_message(raw.message_id, channel=channel)
        user = message.author
        old_reactions = message.reactions.copy()
        message.reactions.clear()
        self.dispatch("reaction_clear", message, old_reactions)
        self.add_message(message, files=False, force=True)

    @tracebacksuppressor
    async def init_ready(self):
        attachments = (file for file in sorted(set(file for file in os.listdir("cache") if file.startswith("attachment_"))))
        for file in attachments:
            with tracebacksuppressor:
                self.attachment_from_file(file)
        print("Loading imported modules...")
        # Wait until all modules have been loaded successfully
        while self.modload:
            fut = self.modload.popleft()
            with tracebacksuppressor:
                # print(fut)
                mod = await fut
        print(f"Mapped command count: {len(self.commands)}")
        commands = set()
        for command in self.commands.values():
            commands.update(command)
        print(f"Unique command count: {len(commands)}")
        # Assign all bot database events to their corresponding keys.
        for u in self.data.values():
            for f in dir(u):
                if f.startswith("_") and f[-1] == "_" and f[1] != "_":
                    func = getattr(u, f, None)
                    if callable(func):
                        self.events.append(f, func)
        print(f"Database event count: {sum(len(v) for v in self.events.values())}")
        await self.fetch_user(self.deleted_user)
        create_task(self.global_loop())
        create_task(self.slow_loop())
        create_task(self.lazy_loop())
        print("Update loops initiated.")
        futs = alist()
        futs.add(create_future(self.update_slash_commands, priority=True))
        futs.add(create_task(self.create_main_website(first=True)))
        futs.add(self.audio_client_start)
        await self.wait_until_ready()
        self.bot_ready = True
        # Send bot_ready event to all databases.
        await self.send_event("_bot_ready_", bot=self)
        for fut in futs:
            with tracebacksuppressor:
                await fut
        print("Bot ready.")
        await wrap_future(self.connect_ready)
        print("Connect ready.")
        self.ready = True
        await create_future(self.update_usernames)
        # Send ready event to all databases.
        await self.send_event("_ready_", bot=self)
        print("Database ready.")
        await self.guilds_ready
        await create_future(self.update_usernames)
        print("Guilds ready.")
        create_task(self.heartbeat_loop())
        force_kill(self.heartbeat_proc)
        create_task(self.fast_loop())
        self.initialisation_complete = True
        print("Initialisation complete.")

    def set_client_events(self):

        print("Setting client events...")

        # The event called when the client first connects, starts initialisation of the other modules
        @self.event
        async def on_connect():
            print("Successfully connected as " + str(self.user))
            self.invite = f"https://discordapp.com/oauth2/authorize?permissions=8&client_id={self.id}&scope=bot%20applications.commands"
            self.mention = (user_mention(self.id), user_pc_mention(self.id))
            if not self.started:
                self.started = True
                create_task(self.init_ready())
            else:
                print("Reconnected.")
            await self.handle_update()

        # The event called when the discord.py state is fully ready.
        @self.event
        async def on_ready():
            self.guilds_ready = create_task(self.load_guilds())
            create_task(aretry(self.get_ip, delay=20))
            await create_future(self.update_subs, priority=True)
            self.update_cache_feed()
            with tracebacksuppressor:
                create_task(self.get_state())
                for guild in self.guilds:
                    if guild.unavailable:
                        print(f"Warning: Guild {guild.id} is not available.")
                await self.handle_update()
            try:
                self.connect_ready.set_result(True)
            except concurrent.futures.InvalidStateError:
                pass

        # Server join message
        @self.event
        async def on_guild_join(guild):
            print(f"New server: {guild}")
            guild = await self.fetch_guild(guild.id)
            self.sub_guilds[guild.id] = guild
            m = guild.me
            await self.send_event("_join_", user=m, guild=guild)
            channel = self.get_first_sendable(guild, m)
            emb = discord.Embed(colour=discord.Colour(8364031))
            emb.set_author(**get_author(self.user))
            emb.description = f"```callback-fun-wallet-{utc()}-\nHi there!```I'm {self.name}, a multipurpose discord bot created by <@201548633244565504>. Thanks for adding me"
            user = None
            if guild.me.guild_permissions.view_audit_log:
                with suppress(discord.Forbidden):
                    a = guild.audit_logs(limit=5, action=discord.AuditLogAction.bot_add)
                    async for e in a:
                        if e.target.id == self.id:
                            user = e.user
                            break
            if user is not None:
                emb.description += f", {user_mention(user.id)}"
                if "dailies" in self.data:
                    self.data.dailies.progress_quests(user, "invite")
            emb.description += (
                f"!\nMy default prefix is `{self.prefix}`, which can be changed as desired on a per-server basis. Mentioning me also serves as an alias for all prefixes.\n"
                + f"For more information, use the `{self.prefix}help` command, "
                + (f"I have a website at [`{self.webserver}`]({self.webserver}), " if self.webserver else "")
                + f"and my source code is available at [`{self.github}`]({self.github}) for those who are interested.\n"
                + "Pleased to be at your service 🙂"
            )
            if not m.guild_permissions.administrator:
                emb.add_field(name="Psst!", value=(
                    "I noticed you haven't given me administrator permissions here.\n"
                    + "That's completely understandable if intentional, but please note that some features may not function well, or not at all, without the required permissions."
                ))
            message = await channel.send(embed=emb)
            await message.add_reaction("✅")
            await self.load_guild_http(guild)
            for member in guild.members:
                name = str(member)
                self.usernames[name] = self.cache.users[member.id]

        # Guild destroy event: Remove guild from bot cache.
        @self.event
        async def on_guild_remove(guild):
            self.users_updated = True
            self.cache.guilds.pop(guild.id, None)
            self.sub_guilds.pop(guild.id, None)
            print("Server lost:", guild, "removed.")

        # Reaction add event: uses raw payloads rather than discord.py message cache. calls _seen_ bot database event.
        @self.event
        async def on_raw_reaction_add(payload):
            try:
                channel = await self.fetch_channel(payload.channel_id)
                user = await self.fetch_user(payload.user_id)
                message = await self.fetch_message(payload.message_id, channel=channel)
            except discord.NotFound:
                return
            emoji = self._upgrade_partial_emoji(payload.emoji)
            if user.id == self.deleted_user:
                print("Deleted user RAW_REACTION_ADD", channel, user, message, emoji, channel.id, message.id)
            await self.seen(user, message.channel, message.guild, event="reaction", raw="Adding a reaction")
            if user.id != self.id:
                if "users" in self.data:
                    self.data.users.add_xp(user, xrand(4, 7))
                    self.data.users.add_gold(user, xrand(1, 5))
                reaction = str(emoji)
                await self.send_event("_reaction_add_", message=message, react=reaction, user=user)
                await self.react_callback(message, reaction, user)
                await self.check_to_delete(message, reaction, user)

        # Reaction remove event: uses raw payloads rather than discord.py message cache. calls _seen_ bot database event.
        @self.event
        async def on_raw_reaction_remove(payload):
            try:
                channel = await self.fetch_channel(payload.channel_id)
                user = await self.fetch_user(payload.user_id)
                message = await self.fetch_message(payload.message_id, channel=channel)
            except discord.NotFound:
                return
            emoji = payload.emoji
            if user.id == self.deleted_user:
                print("Deleted user RAW_REACTION_REMOVE", channel, user, message, emoji, channel.id, message.id)
            await self.seen(user, message.channel, message.guild, event="reaction", raw="Removing a reaction")
            if user.id != self.id:
                reaction = str(emoji)
                await self.react_callback(message, reaction, user)
                await self.check_to_delete(message, reaction, user)

        # Voice state update event: automatically unmutes self if server muted, calls _seen_ bot database event.
        @self.event
        async def on_voice_state_update(member, before, after):
            if member.id == self.id:
                after = member.voice
                if after is not None:
                    if (after.mute or after.deaf) and member.permissions_in(after.channel).deafen_members:
                        # print("Unmuted self in " + member.guild.name)
                        await member.edit(mute=False, deafen=False)
                    await self.handle_update()
            # Check for users with a voice state.
            if after is not None and not after.afk:
                if before is None:
                    if "users" in self.data:
                        self.data.users.add_xp(after, xrand(6, 12))
                        self.data.users.add_gold(after, xrand(2, 5))
                    await self.seen(member, member.guild, event="misc", raw=f"Joining a voice channel")
                elif any((getattr(before, attr) != getattr(after, attr) for attr in ("self_mute", "self_deaf", "self_stream", "self_video"))):
                    await self.seen(member, member.guild, event="misc", raw=f"Updating their voice settings")

        # Typing event: calls _typing_ and _seen_ bot database events.
        @self.event
        async def on_typing(channel, user, when):
            await self.send_event("_typing_", channel=channel, user=user)
            if user.id == self.deleted_user:
                print("Deleted user TYPING", channel, user, channel.id)
            await self.seen(user, getattr(channel, "guild", None), delay=10, event="typing", raw="Typing")

        # Message send event: processes new message. calls _send_ and _seen_ bot database events.
        @self.event
        async def on_message(message):
            self.add_message(message, force=True)
            guild = message.guild
            if guild:
                create_task(self.send_event("_send_", message=message))
            user = message.author
            channel = message.channel
            if user.id == self.deleted_user:
                print("Deleted user MESSAGE", channel, user, message, channel.id, message.id)
            await self.seen(user, channel, guild, event="message", raw="Sending a message")
            await self.react_callback(message, None, user)
            await self.handle_message(message, False)

        # Socket response event: if the event was an interaction, create a virtual message with the arguments as the content, then process as if it were a regular command.
        @self.event
        async def on_socket_response(data):
            self.activity += 1
            if not data.get("op") and data.get("t") == "INTERACTION_CREATE" and "d" in data:
                try:
                    dt = utc_dt()
                    message = self.GhostMessage()
                    d = data["d"]
                    message.id = int(d["id"])
                    message.slash = d["token"]
                    cdata = d.get("data")
                    if d["type"] == 2:
                        name = cdata["name"].replace(" ", "")
                        args = [i.get("value", "") for i in cdata.get("options", ())]
                        argv = " ".join(i for i in args if i)
                        message.content = "/" + name + " " + argv
                        mdata = d.get("member")
                        if not mdata:
                            mdata = d.get("user")
                        else:
                            mdata = mdata.get("user")
                        author = self._state.store_user(mdata)
                        message.author = author
                        channel = None
                        if "resolved" in cdata:
                            res = cdata.get("resolved", {})
                            for mdata in res.get("users", {}).values():
                                message.content += " " + mdata["id"]
                                user = self._state.store_user(mdata)
                            # for m_id in res.get("members", ()):
                            #     message.content += " " + m_id
                            #     user = await self.fetch_user(m_id)
                            for mdata in res.get("messages", {}).values():
                                msg = self.ExtendedMessage.new(mdata)
                                message.content += " " + msg.jump_url
                                self.add_message(msg, force=True)
                                message.channel = msg.channel or message.channel
                        try:
                            channel = self.force_channel(d["channel_id"])
                            guild = await self.fetch_guild(d["guild_id"])
                            message.guild = guild
                            author = guild.get_member(author.id)
                            if author:
                                message.author = author
                        except KeyError:
                            if author is None:
                                raise
                            if channel is None:
                                channel = await self.get_dm(author)
                        if not getattr(message, "guild", None):
                            message.guild = getattr(channel, "guild", None)
                        message.content = single_space(message.content.strip())
                        message.channel = channel
                        message.noref = True
                        await self.process_message(message, msg=message.content, slash=True)
                    elif d["type"] == 3:
                        custom_id = cdata.get("custom_id", "")
                        if "?" in custom_id:
                            custom_id = custom_id.rsplit("?", 1)[0]
                        if custom_id.startswith("▪️"):
                            return await self.ignore_interaction(message)
                        mdata = d.get("member")
                        if not mdata:
                            mdata = d.get("user")
                        else:
                            mdata = mdata.get("user")
                        user = self._state.store_user(mdata)
                        if not user:
                            user = self.GhostUser()
                        channel = None
                        try:
                            channel = self.force_channel(d["channel_id"])
                            guild = await self.fetch_guild(d["guild_id"])
                            user = guild.get_member(user.id)
                        except KeyError:
                            if user is None:
                                raise
                            if channel is None:
                                channel = await self.get_dm(user)
                        message.channel = channel
                        if custom_id.startswith("\x7f"):
                            custom_id = cdata.get("values") or custom_id
                            if type(custom_id) is list:
                                custom_id = " ".join(custom_id)
                        if type(custom_id) is str and custom_id.startswith("~"):
                            m_id, custom_id = custom_id[1:].split("~", 1)
                            custom_id = "~" + custom_id
                        else:
                            m_id = d["message"]["id"]
                        m = await self.fetch_message(m_id, channel)
                        if type(m) is not self.ExtendedMessage:
                            m = self.ExtendedMessage(m)
                            self.add_message(m, force=True)
                        m.int_id = message.id
                        m.int_token = message.slash
                        # print(custom_id, user)
                        await self.react_callback(m, custom_id, user)
                    else:
                        print("Unknown interaction:\n" + str(data))
                    for attr in ("slash", "int_id", "int_token"):
                        try:
                            delattr(message, attr)
                        except AttributeError:
                            pass
                except:
                    print_exc()
                    print("Failed interaction:\n" + str(data))

        # Message edit event: processes edited message, uses raw payloads rather than discord.py message cache. calls _edit_ and _seen_ bot database events.
        @self.event
        async def on_raw_message_edit(payload):
            data = payload.data
            m_id = int(data["id"])
            raw = False
            if payload.cached_message:
                before = payload.cached_message
                after = await self.fetch_message(m_id, payload.channel_id)
            else:
                try:
                    before = await self.fetch_message(m_id)
                except LookupError:
                    # If message was not in cache, create a ghost message object to represent old message.
                    c_id = data.get("channel_id")
                    if not c_id:
                        return
                    before = self.GhostMessage()
                    before.channel = channel = self.force_channel(c_id)
                    before.guild = guild = getattr(channel, "guild", None)
                    before.id = payload.message_id
                    try:
                        u_id = data["author"]["id"]
                    except KeyError:
                        u_id = None
                        before.author = None
                    else:
                        if guild is not None:
                            user = guild.get_member(u_id)
                        else:
                            user = None
                        if not user:
                            user = self._state.store_user(data["author"])
                            # user = await self.fetch_user(u_id)
                        before.author = user
                    try:
                        after = await discord.abc.Messageable.fetch_message(channel, before.id)
                    except LookupError:
                        after = copy.copy(before)
                        after._update(data)
                    else:
                        before.author = after.author
                    raw = True
                else:
                    if type(before) is self.CachedMessage:
                        before = copy.copy(before)
                    after = copy.copy(before)
                    after._update(data)
            with suppress(AttributeError):
                before.deleted = True
            if after.channel is None:
                after.channel = self.force_channel(payload.channel_id)
            after.sem = getattr(before, "sem", None)
            if not after.sem and after.edited_at and (utc_ddt() - after.created_at).total_seconds() >= 3590:
                after.sem = Semaphore(3, 1, rate_limit=20.09)
                async with after.sem:
                    pass
            self.add_message(after, files=False, force=2)
            if before.author.id == self.deleted_user or after.author.id == self.deleted_user:
                print("Deleted user RAW_MESSAGE_EDIT", after.channel, before.author, after.author, before, after, after.channel.id, after.id)
            if raw or before.content != after.content:
                if "users" in self.data:
                    self.data.users.add_xp(after.author, xrand(1, 4))
                if getattr(after, "guild", None):
                    fut = create_task(self.send_event("_edit_", before=before, after=after))
                else:
                    fut = None
                await self.seen(after.author, after.channel, after.guild, event="message", raw="Editing a message")
                if fut:
                    with tracebacksuppressor:
                        await fut
                await self.handle_message(after)

        # Message delete event: uses raw payloads rather than discord.py message cache. calls _delete_ bot database event.
        @self.event
        async def on_raw_message_delete(payload):
            try:
                message = payload.cached_message
                if not message:
                    raise LookupError
            except:
                channel = await self.fetch_channel(payload.channel_id)
                try:
                    message = await self.fetch_message(payload.message_id, channel)
                    if message is None:
                        raise LookupError
                except:
                    # If message was not in cache, create a ghost message object to represent old message.
                    message = self.GhostMessage()
                    message.channel = channel
                    try:
                        message.guild = channel.guild
                    except AttributeError:
                        message.guild = None
                    message.id = payload.message_id
                    message.author = await self.fetch_user(self.deleted_user)
                    message.author.name = "Unknown User"
                    history = discord.abc.Messageable.history(channel, limit=101, around=message)
                    async def flatten_into_cache(history):
                        messages = await history.flatten()
                        data = {m.id: m for m in messages}
                        create_future_ex(self.cache.messages.update, data)
                    create_task(flatten_into_cache(history))
            try:
                message.deleted = True
            except AttributeError:
                message = self.ExtendedMessage(message)
                self.add_message(message, force=True)
                message.deleted = True
            guild = message.guild
            if guild:
                await self.send_event("_delete_", message=message)

        # Message bulk delete event: uses raw payloads rather than discord.py message cache. calls _bulk_delete_ and _delete_ bot database events.
        @self.event
        async def on_raw_bulk_message_delete(payload):
            try:
                messages = payload.cached_messages
                if not messages or len(messages) < len(payload.message_ids):
                    raise LookupError
            except:
                messages = alist()
                channel = await self.fetch_channel(payload.channel_id)
                for m_id in payload.message_ids:
                    try:
                        message = await self.fetch_message(m_id, channel)
                        if message is None:
                            raise LookupError
                    except:
                        # If message was not in cache, create a ghost message object to represent old message.
                        message = self.GhostMessage()
                        message.channel = channel
                        try:
                            message.guild = channel.guild
                        except AttributeError:
                            message.guild = None
                        message.id = m_id
                        message.author = await self.fetch_user(self.deleted_user)
                    messages.add(message)
            out = deque()
            for message in sorted(messages, key=lambda m: m.id):
                try:
                    message.deleted = True
                except AttributeError:
                    message = self.ExtendedMessage(message)
                    self.add_message(message, force=True)
                    message.deleted = True
                out.append(message)
            messages = alist(out)
            await self.send_event("_bulk_delete_", messages=messages)
            for message in messages:
                guild = getattr(message, "guild", None)
                if guild:
                    await self.send_event("_delete_", message=message, bulk=True)

        @self.event
        async def on_guild_update(before, after):
            await self.send_event("_guild_update_", before=before, after=after)

        # User update event: calls _user_update_ and _seen_ bot database events.
        @self.event
        async def on_user_update(before, after):
            b = str(before)
            a = str(after)
            if b != a:
                self.usernames.pop(b, None)
                self.usernames[a] = after
            await self.send_event("_user_update_", before=before, after=after)
            if before.id == self.deleted_user or after.id == self.deleted_user:
                print("Deleted user USER_UPDATE", before, after, before.id, after.id)
            await self.seen(after, event="misc", raw="Editing their profile")

        # Member update event: calls _member_update_ and _seen_ bot database events.
        @self.event
        async def on_member_update(before, after):
            await self.send_event("_member_update_", before=before, after=after)
            if self.status_changed(before, after):
                # A little bit of a trick to make sure this part is only called once per user event.
                # This is necessary because on_member_update is called once for every member object.
                # By fetching the first instance of a matching member object,
                # this ensures the event will not be called multiple times if the user shares multiple guilds with the bot.
                try:
                    member = self.get_member(after.id)
                except LookupError:
                    member = None
                if member is None or member.guild == after.guild:
                    if self.status_updated(before, after):
                        if "dailies" in self.data:
                            self.data.dailies.progress_quests(after, "status")
                        await self.seen(after, event="misc", raw="Changing their status")
                    elif after.status == discord.Status.offline:
                        await self.send_event("_offline_", user=after)

        # Member join event: calls _join_ and _seen_ bot database events.
        @self.event
        async def on_member_join(member):
            name = str(member)
            self.usernames[name] = self.cache.users[member.id]
            if member.guild.id in self._guilds:
                member.guild._member_count = len(member.guild._members)
                if "guilds" in self.data:
                    self.data.guilds.register(member.guild, force=False)
            await self.send_event("_join_", user=member, guild=member.guild)
            await self.seen(member, member.guild, event="misc", raw=f"Joining a server")

        # Member leave event: calls _leave_ bot database event.
        @self.event
        async def on_member_remove(before):
            after = self.cache.users[before.id] = await self._fetch_user(before.id)
            b = str(before)
            a = str(after)
            if b != a:
                self.usernames.pop(b, None)
                self.usernames[a] = after
                await self.send_event("_user_update_", before=before, after=after)
            if hasattr(before, "_user"):
                before._user = after
            if after.id not in self.cache.members:
                name = str(after)
                self.usernames.pop(name, None)
            if before.guild.id in self._guilds:
                before.guild._members.pop(after.id, None)
                before.guild._member_count = len(before.guild._members)
                if "guilds" in self.data:
                    self.data.guilds.register(before.guild, force=False)
            await self.send_event("_leave_", user=before, guild=before.guild)

        # Channel create event: calls _channel_create_ bot database event.
        @self.event
        async def on_guild_channel_create(channel):
            self.sub_channels[channel.id] = channel
            guild = channel.guild
            if guild:
                await self.send_event("_channel_create_", channel=channel, guild=guild)

        # Channel delete event: calls _channel_delete_ bot database event.
        @self.event
        async def on_guild_channel_delete(channel):
            self.sub_channels.pop(channel.id, None)
            self.cache.channels.pop(channel.id, None)
            guild = channel.guild
            if guild:
                await self.send_event("_channel_delete_", channel=channel, guild=guild)

        # Thread delete event: calls _channel_delete_ bot database event.
        @self.event
        async def on_thread_delete(channel):
            self.sub_channels.pop(channel.id, None)
            self.cache.channels.pop(channel.id, None)
            guild = channel.guild
            if guild:
                await self.send_event("_channel_delete_", channel=channel, guild=guild)

        # Thread update event: calls _thread_update_ bot database event.
        @self.event
        async def on_thread_update(before, after):
            await self.send_event("_thread_update_", before=before, after=after)

        # Webhook update event: updates the bot's webhook cache if there are new webhooks.
        @self.event
        async def on_webhooks_update(channel):
            self.data.webhooks.pop(channel.id, None)

        # User ban event: calls _ban_ bot database event.
        @self.event
        async def on_member_ban(guild, user):
            await self.send_event("_ban_", user=user, guild=guild)


class AudioClientInterface:

    clients = {}
    returns = {}
    written = False
    killed = False
    communicating = None

    def __init__(self):
        self.proc = psutil.Popen([python, "x-audio.py"], cwd=os.getcwd() + "/misc", stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=65536)
        if self.communicating:
            self.communicating.join()
        self.communicating = create_thread(self.communicate)
        with suppress():
            if os.name == "nt":
                self.proc.ionice(psutil.IOPRIO_HIGH)
            else:
                self.proc.ionice(psutil.IOPRIO_CLASS_RT, value=7)
        self.fut = concurrent.futures.Future()

    __bool__ = lambda self: self.written

    @property
    def players(self):
        return bot.data.audio.players

    async def asubmit(self, s, aio=False, ignore=False):
        if self.killed:
            return
        bot.activity += 1
        key = ts_us()
        while key in self.returns:
            key += 1
        self.returns[key] = None
        if type(s) not in (bytes, memoryview):
            s = as_str(s).encode("utf-8")
        if aio:
            s = b"await " + s
        if ignore:
            s = b"!" + s
        out = (b"~", orjson.dumps(key), b"~", base64.b85encode(s), b"\n")
        b = b"".join(out)
        self.returns[key] = concurrent.futures.Future()
        try:
            await wrap_future(self.fut)
            self.proc.stdin.write(b)
            self.proc.stdin.flush()
            resp = await asyncio.wait_for(wrap_future(self.returns[key]), timeout=48)
        except (T0, T1, T2, OSError):
            print("AExpired:", s)
            if self.killed:
                raise
            try:
                self.killed = True
                for auds in tuple(self.players.values()):
                    if auds:
                        with tracebacksuppressor:
                            auds.kill()
                await asyncio.sleep(1)
                futs = deque()
                for guild in bot.client.guilds:
                    futs.append(create_task(guild.change_voice_state(channel=None)))
                for fut in futs:
                    await fut
                if is_strict_running(self.proc):
                    force_kill(self.proc)
                self.players.clear()
                self.clients.clear()
                self.returns.clear()
                print("Restarting audio client...")
                self.__init__()
                if "audio" in bot.data:
                    await bot.data.audio._bot_ready_(bot)
            except:
                print_exc()
                await asyncio.sleep(1)
                with suppress():
                    await bot.close()
                touch(bot.restart)
            finally:
                self.killed = False
            raise
        finally:
            self.returns.pop(key, None)
        return resp

    def submit(self, s, aio=False, ignore=False, timeout=48):
        bot.activity += 1
        key = ts_us()
        while key in self.returns:
            key += 1
        self.returns[key] = None
        if type(s) not in (bytes, memoryview):
            s = as_str(s).encode("utf-8")
        if aio:
            s = b"await " + s
        if ignore:
            s = b"!" + s
        out = (b"~", orjson.dumps(key), b"~", base64.b85encode(s), b"\n")
        b = b"".join(out)
        self.returns[key] = concurrent.futures.Future()
        try:
            self.fut.result()
            self.proc.stdin.write(b)
            self.proc.stdin.flush()
            resp = self.returns[key].result(timeout=timeout)
        except:
            raise
        finally:
            self.returns.pop(key, None)
        return resp

    @tracebacksuppressor
    def communicate(self):
        proc = self.proc
        i = b"~0~Fa\n"
        proc.stdin.write(i)
        proc.stdin.flush()
        while not bot.closed and is_strict_running(proc):
            s = proc.stdout.readline().rstrip()
            if s:
                if s.startswith(b"~"):
                    s = base64.b85decode(s[1:])
                    if s == b"bot.audio.returns[0].set_result(0)":
                        break
                print(as_str(s))
            time.sleep(0.2)
        self.written = True
        self.fut.set_result(self)
        while not bot.closed and is_strict_running(proc):
            s = proc.stdout.readline()
            if not s:
                raise EOFError
            s = s.rstrip()
            if s:
                if s[:1] == b"~":
                    bot.activity += 1
                    c = memoryview(base64.b85decode(s[1:]))
                    if c[:18] == b"bot.audio.returns[":
                        out = Dummy
                        if c[-18:] == b"].set_result(None)":
                            out = None
                        elif c[-18:] == b"].set_result(True)":
                            out = True
                        if out is not Dummy:
                            k = int(c[18:-18])
                            with tracebacksuppressor:
                                self.returns[k].set_result(out)
                            continue
                    create_future_ex(exec_tb, c, bot._globals)
                else:
                    print(as_str(s))

    @tracebacksuppressor
    def kill(self):
        if not is_strict_running(self.proc):
            return
        create_future_ex(self.submit, "await kill()", priority=True).result(timeout=2)
        time.sleep(0.5)
        if is_strict_running(self.proc):
            with tracebacksuppressor(psutil.NoSuchProcess):
                return force_kill(self.proc)


# Queries for searching members
# Order of priority:
"""
ID (Full literal match)
Username + Discriminator (Full literal match)
Username (Full case-insensitive match)
Nickname (Full case-insensitive match)
Username + Discriminator (Full alphanumeric match)
Nickname (Full alphanumeric match)
Username + Discriminator (Starting literal match)
Username (Starting case-insensitive match)
Nickname (Starting case-insensitive match)
Username + Discriminator (Starting alphanumeric match)
Nickname (Starting alphanumeric match)
Username + Discriminator (Substring literal match)
Username (Substring case-insensitive match)
Nickname (Substring case-insensitive match)
Username + Discriminator (Substring alphanumeric match)
Nickname (Substring alphanumeric match)
"""
# Results are automatically sorted by match length, randomized if a tie occurs.

def userQuery1(x):
    yield x

def userIter1(x):
    yield str(x)

def userQuery2(x):
    yield str(x).casefold()

def userIter2(x):
    yield str(x)
    yield str(x.name).casefold()
    if getattr(x, "nick", None):
        yield str(x.nick).casefold()

def userQuery3(x):
    yield full_prune(x)

def userIter3(x):
    yield full_prune(x.name)
    if getattr(x, "nick", None):
        yield full_prune(x.nick)

def userQuery4(x):
    yield to_alphanumeric(x).replace(" ", "").casefold()

def userIter4(x):
    yield to_alphanumeric(x.name).replace(" ", "").casefold()
    if getattr(x, "nick", None):
        yield to_alphanumeric(x.nick).replace(" ", "").casefold()


PORT = AUTH.get("webserver_port", 80)
IND = "\x7f"

def update_file_cache():
    attachments = {t for t in bot.cache.attachments.items() if type(t[-1]) is bytes}
    while len(attachments) > 512:
        a_id = next(iter(attachments))
        self.cache.attachments[a_id] = a_id
        attachments.discard(a_id)

def as_file(file, filename=None, ext=None, rename=True):
    if rename:
        fn = round(ts_us())
        for fi in os.listdir("saves/filehost"):
            if fi.startswith(f"{IND}{fn}~"):
                fn += 1
        out = str(fn)
    if hasattr(file, "fp"):
        fp = getattr(file, "_fp", file.fp)
        if type(fp) in (str, bytes):
            rename = True
            filename = file.filename or filename
            file = fp
            if type(file) is bytes:
                file = as_str(file)
        else:
            fp.seek(0)
            filename = file.filename or filename
            file = fp.read()
    if issubclass(type(file), bytes):
        with open(f"cache/temp{out}", "wb") as f:
            f.write(file)
        file = f"cache/temp{out}"
        rename = True
    if rename:
        fo = f"cache/{IND}{out}~.temp$@{lim_str(filename, 64).translate(filetrans)}"
        for i in range(100):
            with suppress(PermissionError):
                os.rename(file, fo)
                break
            time.sleep(0.3)
        n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
        key = base64.urlsafe_b64encode(n.to_bytes(8, "little")).rstrip(b"=").decode("ascii")
        create_task(Request(
            f"http://127.0.0.1:{PORT}/api_register_replacer?ts={out}&key={key}",
            method="PUT",
            aio=True,
            ssl=False,
        ))
    else:
        fn = file.rsplit("/", 1)[-1][1:].rsplit(".", 1)[0].split("~", 1)[0]
    try:
        fn = int(fn)
    except ValueError:
        pass
    else:
        b = fn.bit_length() + 7 >> 3
        fn = as_str(base64.urlsafe_b64encode(fn.to_bytes(b, "big"))).rstrip("=")
    url1 = f"{bot.raw_webserver}/p/{fn}"
    url2 = f"{bot.raw_webserver}/d/{fn}"
    # if filename:
    #     fn = "/" + (str(file) if filename is None else lim_str(filename, 64).translate(filetrans))
    #     url1 += fn
    # if ext and "." not in url1:
    #     url1 += "." + ext
    return url1, url2

def is_file(url):
    for start in (f"{bot.raw_webserver}/", f"http://{bot.ip}:{PORT}/"):
        if url.startswith(start):
            u = url[len(start):]
            endpoint = u.split("/", 1)[0]
            if endpoint in ("view", "file", "files", "download"):
                path = u.split("/", 2)[1].split("?", 1)[0]
                fn = f"{IND}{path}"
                for file in os.listdir("saves/filehost"):
                    if file.rsplit(".", 1)[0].split("~", 1)[0][1:] == path:
                        return f"saves/filehost/{file}"
    return None

def webserver_communicate(bot):
    while not bot.closed:
        while not bot.server:
            time.sleep(12)
        time.sleep(3)
        try:
            assert reqs.next().get(f"http://127.0.0.1:{PORT}/ip", verify=False).content
        except:
            print_exc()
            bot.start_webserver()
            time.sleep(5)
        bot.server_init = True
        with tracebacksuppressor:
            with reqs.next().options(self.webserver, stream=True) as resp:
                self.raw_webserver = resp.url.rstrip("/")
        with tracebacksuppressor:
            while bot.server and is_strict_running(bot.server):
                b = bot.server.stderr.readline()
                if not b:
                    if bot.closed:
                        return
                    bot.start_webserver()
                    break
                b = b.lstrip(b"\x00").rstrip()
                if b:
                    s = as_str(b)
                    if s[0] == "~":
                        create_task(bot.process_http_command(*s[1:].split("\x7f", 3)))
                        bot.activity += 1
                    elif s[0] == "!":
                        create_task(bot.process_http_eval(*s[1:].split("\x7f", 1)))
                        bot.activity += 1
                    elif s == "@@@":
                        bot.activity += 2
                    else:
                        print(s)
            time.sleep(1)
        time.sleep(0.1)


class SimulatedMessage:

    def __init__(self, bot, content, t, name, nick=None, recursive=True):
        self._state = bot._state
        self.created_at = datetime.datetime.utcfromtimestamp(int(t) / 1e6)
        self.ip = name
        self.id = time_snowflake(self.created_at, high=True) - 1
        self.content = content
        self.response = deque()
        if recursive:
            author = self.__class__(bot, content, (ip2int(name) + MIZA_EPOCH) * 1000 + 1, name, nick, recursive=False)
            author.response = self.response
            author.message = self
            author.dm_channel = author
        else:
            author = self
        self.author = author
        self.channel = self
        self.is_channel = True
        self.guild = author
        self.dm_channel = author
        self.name = name
        disc = str(xrand(10000))
        self.discriminator = "0" * (4 - len(disc)) + disc
        self.nick = self.display_name = nick or name
        self.owner_id = self.id
        self.mention = f"<@{self.id}>"
        self.recipient = author
        self.me = bot.user
        self.channels = self.text_channels = self.voice_channels = [author]
        self.members = self._members = [author, bot.user]
        self.message = self
        self.owner = author

    system_content = clean_content = ""
    display_avatar = avatar_url = icon_url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/b9573a17-63e8-4ec1-9c97-2bd9a1e9b515/de1q8lu-eae6a001-6463-4abe-b23c-fc32111c6499.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOiIsImlzcyI6InVybjphcHA6Iiwib2JqIjpbW3sicGF0aCI6IlwvZlwvYjk1NzNhMTctNjNlOC00ZWMxLTljOTctMmJkOWExZTliNTE1XC9kZTFxOGx1LWVhZTZhMDAxLTY0NjMtNGFiZS1iMjNjLWZjMzIxMTFjNjQ5OS5wbmcifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6ZmlsZS5kb3dubG9hZCJdfQ.eih2c_r4mgWKzZx88GKXOd_5FhCSMSbX5qXGpRUMIsE"
    roles = []
    emojis = []
    mentions = []
    attachments = []
    embeds = []
    position = 0
    bot = False
    ghost = True
    simulated = True
    __str__ = lambda self: self.name

    async def send(self, *args, **kwargs):
        if args:
            try:
                kwargs["content"] = args[0]
            except IndexError:
                kwargs["content"] = ""
        try:
            embed = kwargs.pop("embed")
        except KeyError:
            pass
        else:
            if embed is not None:
                e = embed.to_dict()
                if e:
                    kwargs["embed"] = e
        try:
            embeds = kwargs.pop("embeds")
        except KeyError:
            pass
        else:
            embeds = [e for e in (embed.to_dict() for embed in embeds if embed is not None) if e]
            if embeds:
                kwargs["embeds"] = embeds
        try:
            file = kwargs.pop("file")
        except KeyError:
            pass
        else:
            f = await create_future(as_file, file)
            kwargs["file"] = f[0]
        self.response.append(kwargs)
        return self

    def edit(self, **kwargs):
        self.response[-1].update(kwargs)
        return as_fut(self)

    async def history(self, *args, **kwargs):
        yield self

    get_member = lambda self, *args: None
    delete = async_nop
    add_reaction = async_nop
    delete_messages = async_nop
    trigger_typing = async_nop
    webhooks = lambda self: []
    guild_permissions = discord.Permissions((1 << 32) - 1)
    permissions_for = lambda self, member=None: self.guild_permissions
    permissions_in = lambda self, channel=None: self.guild_permissions
    invites = lambda self: exec("raise LookupError")


async def desktop_identify(self):
    """Sends the IDENTIFY packet."""
    print("Overriding with desktop status...")
    payload = {
        'op': self.IDENTIFY,
        'd': {
            'token': self.token,
            'properties': {
                'os': 'Miza-OS',
                'browser': 'Discord Client',
                'device': 'Miza',
                'referrer': '',
                'referring_domain': ''
            },
            'compress': True,
            'large_threshold': 250,
            'v': 3
        }
    }

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

discord.gateway.DiscordWebSocket.identify = lambda self: desktop_identify(self)


# If this is the module being run and not imported, create a new Bot instance and run it.
if __name__ == "__main__":
    # Redirects all output to the main log manager (PRINT).
    _print = print
    with contextlib.redirect_stdout(PRINT):
        with contextlib.redirect_stderr(PRINT):
            PRINT.start()
            sys.stdout = sys.stderr = print = PRINT
            print("Logging started.")
            create_future_ex(proc_start)
            create_task(Request._init_())
            self = miza = bot = client = BOT[0] = Bot()
            miza.http.user_agent = "Miza"
            miza.miza = miza
            with miza:
                miza.run()
            force_kill(miza.server)
            miza.audio.kill()
            sub_kill(start=False, force=True)
    print = _print
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
