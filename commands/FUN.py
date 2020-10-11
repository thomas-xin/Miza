try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class GameOverError(OverflowError):
    pass


# Represents and manages an N-dimensional game of 2048, with many optional settings.
class ND2048(collections.abc.MutableSequence):

    digit_ratio = 1 / math.log2(10)
    spl = b"_"
    __slots__ = ("data", "history", "shape", "flags")

    # Loads a new instance from serialized data
    @classmethod
    def load(cls, data):
        spl = data.split(cls.spl)
        i = spl.index(b"")
        shape = [int(x) for x in spl[:i - 1]]
        flags = int(spl[i - 1])
        spl = spl[i + 1:]
        self = cls(flags=flags)
        self.data = np.frombuffer(b642bytes(spl.pop(0), 1), dtype=np.int8).reshape(shape)
        if not self.data.flags.writeable:
            self.data = self.data.copy()
        if self.flags & 1:
            self.history = deque(maxlen=max(1, int(800 / np.prod(self.data.size) - 1)))
            while spl:
                self.history.append(np.frombuffer(b642bytes(spl.pop(0), 1), dtype=np.int8).reshape(shape))
        return self

    # Serializes gamestate data to base64
    def serialize(self):
        s = (self.spl.join(str(i).encode("utf-8") for i in self.data.shape)) + self.spl + str(self.flags).encode("utf-8") + self.spl * 2 + bytes2b64(self.data.tobytes(), 1)
        if self.flags & 1 and self.history:
            s += self.spl + self.spl.join(bytes2b64(b.tobytes(), 1) for b in self.history)
        return s

    # Initializes new game
    def __init__(self, *size, flags=0):
        if not size:
            self.data = None
            self.flags = flags
            return
        elif len(size) <= 1:
            try:
                size = int(size)
            except (ValueError, TypeError):
                size = list(reversed(size))
            else:
                size = [size, size]
        self.data = np.tile(np.int8(0), size)
        if flags & 1:
            # Maximum undo steps based on the size of the game board
            self.history = deque(maxlen=max(1, int(800 / np.prod(size) - 1)))
        self.flags = flags
        self.spawn(max(2, self.data.size // 6), flag_override=0)

    __repr__ = lambda self: self.__class__.__name__ + ".load(" + repr(self.serialize()) + ")"

    # Displays game board for dimensions N <= 4
    def __str__(self):
        m = 64
        a = self.data
        nd = max(3, 1 + int(self.digit_ratio * np.max(a)), 2 + int(-self.digit_ratio * np.min(a)))
        w = len(a) * (nd + 1) - 1
        shape = list(a.shape)
        if a.ndim <= 2:
            if a.ndim == 1:
                a = [a]
            else:
                a = np.moveaxis(a, 0, 1)
            return "+" + ("-" * nd + "+") * shape[0] + "\n" + "\n".join("|" + "|".join(self.display(x, nd) for x in i) + "|\n+" + ("-" * nd + "+") * shape[0] for i in a)
        curr = 3
        horiz = 1
        while a.ndim >= curr:
            w2 = shape[a.ndim - curr - 1]
            c = (w + 1) * w2 - 1
            if c <= m:
                w = c
                horiz += 1
            curr += 2
        if a.ndim <= 4:
            dim = len(a)
            if a.ndim == 3:
                a = np.expand_dims(np.rollaxis(a, 0, 3), 0)
                shape.insert(3, 1)
            else:
                a = np.swapaxes(a, 0, 3)
            horiz = 2
            if horiz > 1:
                return "\n".join("\n".join((("+" + ("-" * nd + "+") * shape[0] + " ") * shape[2])[:-1] + "\n" + " ".join("|" + "|".join(self.display(x, nd) for x in i) + "|" for i in j) for j in k) + "\n" + (("+" + ("-" * nd + "+") * shape[0] + " ") * shape[2])[:-1] for k in a)
            return str(horiz)
        return self.data.__str__()

    # Calulates effective total score (value of a tile 2 ^ x is (2 ^ x)(x - 1)
    score = lambda self: np.sum([(g - 1) * (1 << g) for g in [self.data[self.data > 1].astype(object)]])

    # Randomly spawns tiles on a board, based on the game's settings. May be overridden by the flag_override argument.
    def spawn(self, count=1, flag_override=None):
        try:
            flags = flag_override if flag_override is not None else self.flags
            if 0 not in self:
                raise IndexError
            if flags & 4:
                # Scale possible number spawns to highest number on board
                high = max(4, numpy.max(self.data)) - 1
                choices = [np.min(self.data[self.data > 0])] + [max(1, i) for i in range(high - 4, high)]
            else:
                # Default 2048 probabilities: 90% ==> 2, 10% ==> 4
                choices = [1] * 9 + [2]
            if flags & 2:
                # May spawn negative numbers if special tiles mode is on
                neg = max(1, numpy.max(self.data))
                neg = 1 if neg <= 1 else random.randint(1, neg)
                neg = -1 if neg <= 1 else -random.randint(1, neg)
                for i in range(len(choices) >> 2):
                    if neg >= 0:
                        break
                    choices[i + 1] = neg
                    neg += 1
            # Select a list from possible spawns and distribute them into random empty locations on the game board
            spawned = deque(choice(choices) for i in range(count))
            fi = self.data.flat
            empty = [i for i in range(self.data.size) if not fi[i]]
            random.shuffle(empty)
            empty = deque(empty)
            while spawned:
                i = empty.popleft()
                if not fi[i]:
                    fi[i] = spawned.popleft()
        except IndexError:
            raise RuntimeError("Unable to spawn tile.")
        return self

    # Recursively splits the game board across dimensions, then performs a move across each column, returns True if game board was modified
    def recurse(self, it):
        if it.ndim > 1:
            return any([self.recurse(i) for i in it])
        done = modified = False
        while not done:
            done = True
            for i in range(len(it) - 1):
                if it[i + 1]:
                    # If current tile is empty, move next tile into current position
                    if not it[i]:
                        it[i] = it[i + 1]
                        it[i + 1] = 0
                        done = False
                    # If current tile can combine with adjacent tile, add them
                    elif it[i] == it[i + 1] or (it[i + 1] < 0 and it[i]):
                        it[i] += (1 if it[i] >= 0 else -1) * (1 if it[i + 1] >= 0 else -it[i + 1])
                        it[i + 1] = 0
                        done = False
                    elif it[i] < 0:
                        it[i] = it[i + 1] - it[i]
                        it[i + 1] = 0
                        done = False
                if not done:
                    modified = True
        return modified

    # Performs a single move across a single dimension.
    def move(self, dim=0, rev=False, count=1):
        # Creates backup copy of game data if easy mode is on
        if self.flags & 1:
            temp = copy.deepcopy(self.data)
        moved = False
        for i in range(count):
            # Random move selector
            if dim < 0:
                dim = xrand(self.data.ndim)
                rev = xrand(2)
            # Selects a dimension to move against
            it = np.moveaxis(self.data, dim, -1)
            if rev:
                it = np.flip(it)
            m = self.recurse(it)
            if m:
                self.spawn()
            moved |= m
        if moved:
            if self.flags & 1:
                self.history.append(temp)
            # If board is full, attempt a move in both directions of every dimension, and announce game over if none are possible
            if 0 not in self.data:
                valid = False
                for dim in range(self.data.ndim):
                    temp = np.moveaxis(copy.deepcopy(self.data), dim, -1)
                    if self.recurse(temp) or self.recurse(np.flip(temp)):
                        valid = True
                        break
                if not valid:
                    raise GameOverError("Game Over.")
        return moved

    # Attempt to perform an undo
    def undo(self):
        if self.flags & 1 and self.history:
            self.data = self.history.pop()
            return True

    # Creates a display of a single number in a text box of a fixed length, align centre then right
    def display(self, num, length):
        if num > 0:
            numlength = 1 + int(num * self.digit_ratio)
            out = str(1 << int(num))
        elif num < 0:
            numlength = 2 - int(num * self.digit_ratio)
            out = "×" + str(1 << int(-num))
        else:
            return " " * length
        x = length - numlength
        return " " * (1 + x >> 1) + out + " " * (x >> 1)

    __len__ = lambda self: self.data.__len__()
    __iter__ = lambda self: self.data.flat
    __reversed__ = lambda self: np.flip(self.data).flat
    __contains__ = lambda self, *args: self.data.__contains__(*args)
    __getitem__ = lambda self, *args: self.data.__getitem__(*args)
    __setitem__ = lambda self, *args: self.data.__setitem__(*args)
    __delitem__ = lambda self, *args: self.data.__delitem__(*args)
    insert = lambda self, *args: self.data.insert(*args)
    render = lambda self: self.__str__()


class Text2048(Command):
    time_consuming = True
    name = ["2048"]
    min_level = 0
    description = "Plays a game of 2048 using reactions. Gained points are rewarded as gold."
    usage = "<0*:dimension_sizes[4x4]> <1:dimension_count[2]> <special_tiles(?s)> <public(?p)> <insanity_mode(?i)> <easy_mode(?e)>"
    flags = "pies"
    rate_limit = (3, 9)
    reacts = ("⬅️", "➡️", "⬆️", "⬇️", "⏪", "⏩", "⏫", "⏬", "◀️", "▶️", "🔼", "🔽", "👈", "👉", "👆", "👇")
    directions = demap((r.encode("utf-8"), i) for i, r in enumerate(reacts))
    directions[b'\xf0\x9f\x92\xa0'] = -2
    directions[b'\xe2\x86\xa9\xef\xb8\x8f'] = -1

    async def _callback_(self, bot, message, reaction, argv, user, perm, vals, **void):
        # print(user, message, reaction, argv)
        u_id, mode = [int(x) for x in vals.split("_", 1)]
        if reaction is not None and u_id != user.id and u_id != 0 and perm < 3:
            return
        spl = argv.split("-")
        size = [int(x) for x in spl.pop(0).split("_")]
        data = None
        if reaction is None:
            # If game has not been started, add reactions and create new game
            for react in self.directions.a:
                r = self.directions.a[react]
                if r == -2 or (r == -1 and mode & 1) or r >= 0 and r >> 1 < len(size):
                    create_task(message.add_reaction(react.decode("utf-8")))
            g = ND2048(*size, flags=mode)
            data = g.serialize()
            score = 0
        else:
            # Get direction of movement
            data = "-".join(spl).encode("utf-8")
            reac = reaction
            if reac not in self.directions:
                return
            r = self.directions[reac]
            score = 0
            try:
                # Undo action only works in easy mode
                if r == -1:
                    if not mode & 1:
                        return
                    g = ND2048.load(data)
                    if not g.undo():
                        return
                    data = g.serialize()
                # Random moves
                elif r == -2:
                    g = ND2048.load(data)
                    if not g.move(-1, count=16):
                        return
                    data = g.serialize()
                # Regular moves; each dimension has 2 possible moves
                elif r >> 1 < len(size):
                    g = ND2048.load(data)
                    score = g.score()
                    if not g.move(r >> 1, r & 1):
                        return
                    data = g.serialize()
            except GameOverError:
                if u_id == 0:
                    u = None
                elif user.id == u_id:
                    u = user
                else:
                    u = bot.get_user(u_id, replace=True)
                emb = discord.Embed(colour=discord.Colour(1))
                if u is None:
                    emb.set_author(name="@everyone", icon_url=bot.discord_icon)
                else:
                    emb.set_author(**get_author(u))
                emb.description = ("**```fix\n" if mode & 6 else "**```\n") + g.render() + "```**"
                fscore = g.score()
                if score is not None:
                    xp = max(0, fscore - score) * 16 / np.prod(g.data.shape)
                    if mode & 1:
                        xp /= math.sqrt(2)
                    elif mode & 2:
                        xp /= 2
                    elif mode & 4:
                        xp /= 3
                    bot.data.users.add_gold(user, xp)
                    emb.description += await bot.as_rewards(f"+{int(xp)}")
                emb.set_footer(text=f"Score: {fscore}")
                # Clear reactions and announce game over message
                await message.edit(content="**```\n2048: GAME OVER```**", embed=emb)
                if message.guild and message.guild.get_member(bot.client.user.id).permissions_in(message.channel).manage_messages:
                    await message.clear_reactions()
                else:
                    for reaction in message.reactions:
                        if reaction.me:
                            await message.remove_reaction(reaction.emoji, bot.client.user if message.guild is None else message.guild.get_member(bot.client.user.id))
                for c in ("🇬", "🇦", "🇲", "🇪", "⬛", "🇴", "🇻", "3️⃣", "🇷"):
                    create_task(message.add_reaction(c))
                    await asyncio.sleep(0.5)
                return
        if data is not None:
            # Update message if gamestate has been changed
            if u_id == 0:
                u = None
            elif user.id == u_id:
                u = user
            else:
                u = bot.get_user(u_id, replace=True)
            emb = discord.Embed(colour=rand_colour())
            if u is None:
                emb.set_author(name="@everyone", icon_url=bot.discord_icon)
            else:
                emb.set_author(**get_author(u))
            content = "*```callback-fun-text2048-" + str(u_id) + "_" + str(mode) + "-" + "_".join(str(i) for i in size) + "-" + data.decode("utf-8") + "\nPlaying 2048...```*"
            emb.description = ("**```fix\n" if mode & 6 else "**```\n") + g.render() + "```**"
            fscore = g.score()
            if score is not None:
                xp = max(0, fscore - score) * 16 / np.prod(g.data.shape)
                if mode & 1:
                    xp /= math.sqrt(2)
                elif mode & 2:
                    xp /= 2
                elif mode & 4:
                    xp /= 3
                bot.data.users.add_gold(user, xp)
                emb.description += await bot.as_rewards(f"+{int(xp)}")
            emb.set_footer(text=f"Score: {fscore}")
            await message.edit(content=content, embed=emb)

    async def __call__(self, bot, argv, args, user, flags, guild, **void):
        # Input may be nothing, a single value representing board size, a size and dimension count input, or a sequence of numbers representing size along an arbitrary amount of dimensions
        if not len(argv.replace(" ", "")):
            size = [4, 4]
        else:
            if "x" in argv:
                size = await recursive_coro([bot.eval_math(i, user) for i in argv.split("x")])
            else:
                if len(args) > 1:
                    dims = args.pop(-1)
                    dims = await bot.eval_math(dims, user)
                else:
                    dims = 2
                if dims <= 0:
                    raise ValueError("Invalid amount of dimensions specified.")
                width = await bot.eval_math(" ".join(args), user)
                size = list(repeat(width, dims))
        if len(size) > 8:
            raise OverflowError("Board size too large.")
        items = 1
        for x in size:
            items *= x
            if items > 256:
                raise OverflowError("Board size too large.")
        # Prepare game settings, send callback message to schedule game start
        mode = 0
        if "p" in flags:
            u_id = 0
            mode |= 8
        else:
            u_id = user.id
        if "i" in flags:
            mode |= 4
        if "s" in flags:
            mode |= 2
        if "e" in flags:
            mode |= 1
        return "*```callback-fun-text2048-" + str(u_id) + "_" + str(mode) + "-" + "_".join(str(i) for i in size) + "\nStarting Game...```*"

    
class SlotMachine(Command):
    name = ["Slots"]
    min_level = 0
    description = "Plays a slot machine game. Costs gold to play, can yield gold and diamonds."
    usage = "<bet[100]> <skip_animation(?s)>"
    flags = "s"
    rate_limit = (5, 10)
    emojis = {
        "❤️": 20,
        "🍒": 6,
        "💎": None,
        "🍎": 4,
        "🍇": 5,
        "🍋": 2,
        "🍉": 1,
        "🍌": 3,
    }

    def select(self):
        x = random.random()
        if x < 1 / 32:
            return "💎"
        elif x < 3 / 32:
            return "❤️"
        return choice("🍒🍎🍇🍋🍉🍌")

    def generate(self, count=3):
        x = random.random()
        if x < 1 / 5:
            count = 3
        elif x < 8 / 15:
            count = 2
        else:
            count = 1
        out = alist((self.select(),) * count)
        while len(out) < 3:
            out.append(choice(self.emojis))
        return shuffle(out)

    async def as_emojis(self, wheel):
        out = ""
        for item in wheel:
            if item is None:
                out += await create_future(self.bot.data.emojis.emoji_as, "slot_machine.gif")
            else:
                out += item
        return out

    async def __call__(self, argv, user, flags, **void):
        b1 = 5
        b2 = 50
        if argv:
            bet = await self.bot.eval_math(argv, user)
            if bet < b1:
                raise ValueError(f"Minimum bet is {b1} coins.")
            if bet > b2:
                raise ValueError(f"Maximum bet is {b2} coins.")
        else:
            bet = b1
        if bet > self.bot.data.users.get(user.id, {}).get("gold", 0):
            raise OverflowError("Bet cannot be greater than your balance.")
        self.bot.data.users.add_gold(user, -bet)
        skip = int("s" in flags)
        return f"*```callback-fun-slotmachine-{user.id}_{bet}_{skip}-\nLoading Slot Machine...```*"

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        spl = [int(i) for i in vals.split("_", 2)]
        if len(spl) < 3:
            spl.append(0)
        u_id, bet, skip = spl
        if reaction is None or reaction.decode("utf-8", "replace") == "⤵️":
            if reaction is None:
                create_task(message.add_reaction("⤵️"))
                user = await bot.fetch_user(u_id)
            else:
                if bet > bot.data.users.get(user.id, {}).get("gold", 0):
                    raise OverflowError("Bet cannot be greater than your balance.")
                bot.data.users.add_gold(user, -bet)
            wheel_true = self.generate()
            wheel_display = [None] * 3 if not skip else wheel_true
            wheel_order = deque(shuffle(range(3))) if not skip else deque((0, ))
            emb = discord.Embed(colour=rand_colour()).set_author(**get_author(user))
            if not skip:
                async with delay(2):
                    emoj = await self.as_emojis(wheel_display)
                    gold = bot.data.users.get(user.id, {}).get("gold", 0)
                    bets = await bot.as_rewards(bet)
                    bals = await bot.as_rewards(gold)
                    emb.description = f"```css\n[Slot Machine]```{emoj}\nBet: {bets}\nBalance: {bals}"
                    await message.edit(content=None, embed=emb)
            ctx = delay(1) if not skip else emptyctx
            while wheel_order:
                async with ctx:
                    i = wheel_order.popleft()
                    wheel_display[i] = wheel_true[i]
                    if not wheel_order:
                        gold = diamonds = 0
                        start = f"```callback-fun-slotmachine-{user.id}_{bet}-\n"
                        if wheel_true[0] == wheel_true[1] == wheel_true[2]:
                            gold = self.emojis[wheel_true[0]]
                            if gold is None:
                                diamonds = bet / 5
                            else:
                                gold *= bet
                        bot.data.users.add_diamonds(user, diamonds)
                        bot.data.users.add_gold(user, gold)
                        rewards = await bot.as_rewards(diamonds, gold)
                        end = f"\nRewards:\n{rewards}\n"
                    else:
                        start = "```ini\n"
                        end = ""
                    emoj = await self.as_emojis(wheel_display)
                    gold = bot.data.users.get(user.id, {}).get("gold", 0)
                    bets = await bot.as_rewards(bet)
                    bals = await bot.as_rewards(gold)
                    emb.description = f"{start}[Slot Machine]```{emoj}\nBet: {bets}\nBalance: {bals}{end}"
                    await message.edit(embed=emb)


class Pay(Command):
    name = ["GiveCoins", "GiveGold"]
    min_level = 0
    description = "Pays a specified amount of coins to the target user."
    usage = "<0:user> <1:amount[1]>"
    rate_limit = 0.5

    async def __call__(self, bot, user, args, guild, **void):
        if not args:
            raise ArgumentError("Please input target user.")
        target = await bot.fetch_user_member(args.pop(0), guild)
        if target.id == bot.id:
            return "\u200bI appreciate the generosity, but I have enough already :3"
        if args:
            amount = await bot.eval_math(" ".join(args), user)
        else:
            amount = 1
        if amount <= 0:
            raise ValueError("Please input a valid amount of coins.")
        if not amount <= bot.data.users.get(user.id, {}).get("gold", 0):
            raise OverflowError("Payment cannot be greater than your balance.")
        bot.data.users.add_gold(user, -amount)
        bot.data.users.add_gold(target, amount)
        bot.data.dailies.progress_quests(user, "pay", amount)
        return css_md(f"{sqr_md(user)} has paid {sqr_md(amount)} coins to {sqr_md(target)}.")


class Dogpile(Command):
    server_only = True
    min_level = 2
    description = "Causes ⟨MIZA⟩ to automatically imitate users when 3+ of the same messages are posted in a row. Grants XP and gold when triggered."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"
    rate_limit = 0.5

    async def __call__(self, flags, guild, **void):
        update = self.data.dogpiles.update
        bot = self.bot
        following = bot.data.dogpiles
        curr = following.get(guild.id, False)
        if "d" in flags:
            if guild.id in following:
                following.pop(guild.id)
                update()
            return css_md(f"Disabled dogpile imitating for {sqr_md(guild)}.")
        if "e" in flags or "a" in flags:
            following[guild.id] = True
            update()
            return css_md(f"Enabled dogpile imitating for {sqr_md(guild)}.")
        if curr:
            return ini_md(f"Dogpile imitating is currently enabled in {sqr_md(guild)}.")
        return ini_md(f"Dogpile imitating is currently disabled in {sqr_md(guild)}. Use ?e to enable.")


class UpdateDogpiles(Database):
    name = "dogpiles"

    def __load__(self):
        self.msgFollow = cdict()

    async def _nocommand_(self, text, edit, orig, message, **void):
        if message.guild is None or not orig:
            return
        g_id = message.guild.id
        following = self.data
        if g_id in following:
            u_id = message.author.id
            c_id = message.channel.id
            if not edit:
                if following[g_id]:
                    checker = orig
                    curr = self.msgFollow.get(c_id)
                    if curr is None:
                        curr = [checker, 1, 0]
                        self.msgFollow[c_id] = curr
                    # Must not imitate same user spamming
                    elif checker == curr[0] and u_id != curr[2]:
                        curr[1] += 1
                        if curr[1] >= 3:
                            curr[1] = xrand(-3) + 1
                            if len(checker):
                                create_task(message.channel.send(checker, tts=message.tts))
                                self.bot.data.users.add_xp(message.author, len(message.content) / 2 + 16)
                                self.bot.data.users.add_gold(message.author, len(message.content) / 4 + 32)
                    else:
                        # Don't imitate messages longer than 128 characters to prevent spam
                        if len(checker) > 128:
                            checker = ""
                        curr[0] = checker
                        curr[1] = xrand(-1, 2)
                    curr[2] = u_id


class Daily(Command):
    name = ["Quests", "Quest", "Tasks", "Challenges", "Dailies"]
    min_level = 0
    description = "Shows your list of daily quests."
    rate_limit = 1

    async def __call__(self, bot, user, channel, **void):
        data = bot.data.dailies.get(user)
        emb = discord.Embed(title="Daily Quests", colour=rand_colour()).set_author(**get_author(user))
        bal = await bot.data.users.get_balance(user)
        c = len(data['quests'])
        emb.description = f"```callback-fun-daily-{user.id}-\n{c} task{'' if c == 1 else 's'} available\n{sec2time(86400 - utc() + data['time'])} remaining```Balance: {bal}"
        for field in data["quests"][:5]:
            bar = await bot.create_progress_bar(10, field.progress / field.required)
            rewards = await bot.as_rewards(field.get("diamonds", None), field.get("gold", None))
            emb.add_field(name=field.name, value=f"{bar} `{int(field.progress)}/{field.required}`\nRewards: {rewards}", inline=False)
        message = await channel.send(embed=emb)
        create_task(message.add_reaction("✅"))
    
    async def _callback_(self, bot, user, reaction, message, perm, vals, **void):
        if reaction is None:
            return
        if reaction.decode("utf-8", "replace") != "✅":
            return
        u_id = vals
        if str(user.id) != u_id:
            return
        data = bot.data.dailies.collect(user)
        emb = discord.Embed(title="Daily Quests", colour=rand_colour()).set_author(**get_author(user))
        bal = await bot.data.users.get_balance(user)
        c = len(data['quests'])
        emb.description = f"```callback-fun-daily-{user.id}-\n{c} task{'' if c == 1 else 's'} available\n{sec2time(86400 - utc() + data['time'])} remaining```Balance: {bal}"
        for field in data["quests"][:5]:
            bar = await bot.create_progress_bar(10, field.progress / field.required)
            rewards = await bot.as_rewards(field.get("diamonds", None), field.get("gold", None))
            emb.add_field(name=field.name, value=f"{bar} `{floor(field.progress)}/{field.required}`\nRewards: {rewards}", inline=False)
        return await message.edit(embed=emb)


class UpdateDailies(Database):
    name = "dailies"
    user = True

    def __load__(self, **void):
        self.typing = {}

    def get(self, user):
        data = self.data.get(user.id)
        if data is None or utc() - data["time"] >= 86400:
            data = self.data[user.id] = dict(quests=self.generate(user), time=zerot())
            self.update()
        return data

    def collect(self, user):
        data = self.get(user)
        quests = data["quests"]
        pops = set()
        for i, quest in enumerate(quests):
            if quest.progress >= quest.required:
                self.bot.data.users.add_diamonds(user, quest.get("diamonds"))
                self.bot.data.users.add_gold(user, quest.get("gold"))
                pops.add(i)
        if pops:
            quests.pops(pops)
            self.update()
        return data

    def generate(self, user):
        if user.id == self.bot.id or self.bot.is_blacklisted(user.id):
            return dict(quests=(), time=inf)
        level = self.bot.data.users.xp_to_level(self.bot.data.users.get_xp(user))
        quests = alist()
        for i in range(min(20, level + 5 >> 1)):
            q_id = xrand(11)
            if q_id == 0:
                x = round((level * 10 + 100) * random.random() + 70)
                q = cdict(name=f"Post {x} messages", gold=x * 2, progress=0, required=x, action="send")
            elif q_id == 1:
                q = cdict(name=f"Invite me to a server and/or react to the join message", diamonds=floor(50 + level * 2 / 3), progress=0, required=1, action="invite")
            elif q_id == 2:
                q = cdict(name=f"Earn 1 diamond", gold=level * 50, progress=0, required=1, action="diamond")
            elif q_id == 3:
                x = round((level + 10) * random.random() + 20)
                q = cdict(name=f"Use {x} commands", gold=x * 9, progress=0, required=x, action="command")
            elif q_id == 4:
                x = round((level * 200 + 1000) * random.random() + 1000)
                q = cdict(name=f"Earn {x} experience", gold=x >> 1, progress=0, required=x, action="xp")
            elif q_id == 5:
                x = round((level + 5) * random.random() + 3) * 60
                q = cdict(name=f"Listen to {sec2time(x)} of my music", gold=x, progress=0, required=x, action="music")
            elif q_id == 6:
                x = round((level * 250 + 1000) * random.random() + 2000)
                q = cdict(name=f"Send {x} total characters of text", gold=x >> 3, progress=0, required=x, action="text")
            elif q_id == 7:
                x = round((level * 5 + 20) * random.random() + 20)
                q = cdict(name=f"Add {x} reactions", gold=x * 5, progress=0, required=x, action="react")
            elif q_id == 8:
                x = round((level * 100 + 500) * random.random() + 400)
                q = cdict(name=f"Pay {x} to other users", gold=x >> 1, progress=0, required=x, action="pay")
            elif q_id == 9:
                x = round((level * 28 + 240) * random.random() + 180)
                q = cdict(name=f"Type for {sec2time(x)}", gold=x * 2, progress=0, required=x, action="typing")
            elif q_id == 10:
                x = xrand(10, 21)
                q = cdict(name=f"Talk to me {x} times", gold=x * 7, progress=0, required=x, action="talk")
            quests.append(q)
        return quests.appendleft(cdict(name="Daily rewards", gold=level * 50 + 400, progress=1, required=1, action=None))

    def progress_quests(self, user, action, value=1):
        if user.id == self.bot.id or self.bot.is_blacklisted(user.id):
            return
        data = self.get(user)
        quests = data["quests"]
        for i in range(min(5, len(quests))):
            quest = quests[i]
            if quest.action == action:
                quest.progress += value
                self.update()

    def valid_message(self, message):
        user = message.author
        self.progress_quests(user, "send")
        self.progress_quests(user, "text", get_message_length(message))
        if user.id in self.typing:
            t = utc()
            self.progress_quests(user, "typing", t - self.typing.pop(user.id, t))

    def _command_(self, user, loop=False, **void):
        if not loop:
            self.progress_quests(user, "command")

    def _typing_(self, user, **void):
        if user.id in self.typing:
            if utc() - self.typing[user.id] > 10:
                self.progress_quests(user, "typing", 10)
                self.typing[user.id] = utc()
        else:
            self.typing[user.id] = utc()


class Wallet(Command):
    name = ["Bal", "Balance"]
    min_level = 0
    description = "Shows the target users' wallet."
    usage = "<*objects>"
    rate_limit = 1
    multi = True

    async def __call__(self, bot, args, argv, argl, user, guild, channel, **void):
        users = await bot.find_users(argl, args, user, guild)
        if not users:
            raise LookupError("No results found.")
        for user in users:
            data = bot.data.users.get(user.id, {})
            xp = bot.data.users.get_xp(user)
            level = bot.data.users.xp_to_level(xp)
            xp_curr = bot.data.users.xp_required(level)
            xp_next = bot.data.users.xp_required(level + 1)
            ratio = (xp - xp_curr) / (xp_next - xp_curr)
            gold = data.get("gold", 0)
            diamonds = data.get("diamonds", 0)
            bar = await bot.create_progress_bar(18, ratio)
            xp = floor(xp)
            bal = await bot.as_rewards(diamonds, gold)
            description = f"{bar}\n`Lv {level}`\n`XP {xp}/{xp_next}`\n{bal}"
            bot.send_as_embeds(channel, description, thumbnail=best_url(user), author=get_author(user), colour=xrand(1536))

    join_cache = {}

    async def _callback_(self, bot, message, reaction, user, vals, **void):
        ts = int(float(vals))
        if utc() - ts > 86400:
            self.join_cache.pop(message.id, None)
            return
        if reaction is None or reaction.decode("utf-8", "replace") != "✅":
            return
        cache = set_dict(self.join_cache, message.id, set())
        if len(cache) > 256:
            cache.pop()
        if user.id in cache:
            return
        cache.add(user.id)
        bot.data.dailies.progress_quests(user, "invite")


class Shop(Command):
    min_level = 0
    description = "Displays the shop system, or purchases an item."
    usage = "<item[]>"
    rate_limit = 1

    products = cdict(
        upgradeserver=cdict(
            name="Upgrade Server",
            cost=[1728, 248832],
            description="Upgrades the server's privilege level, granting access to all command categories and reducing command cooldown.",
        ),
    )

    async def __call__(self, bot, guild, channel, user, argv, **void):
        if not argv:
            desc = ""
            for product in self.products.values():
                cost = await bot.as_rewards(*product.cost)
                description = ini_md(f"{sqr_md(product.name)} {cost}\n{product.description}")
            return bot.send_as_embeds(channel, description, title="Shop", author=get_author(user))
        item = argv.replace("-", "").replace("_", "").replace(" ", "").casefold()
        try:
            product = self.products[item]
        except KeyError:
            raise LookupError(f"Sorry, we don't sell {argv} here...")
        data = bot.data.users.get(user.id, {})
        gold = data.get("gold", 0)
        diamonds = data.get("diamonds", 0)
        if len(product.cost) < 2 or diamonds >= product.cost[0]:
            if gold >= product.cost[-1]:
                if product.name == "Upgrade Server":
                    if hasattr(guild, "ghost"):
                        return "```\nThis item can only be purchased in a server.```"
                    if bot.is_trusted(guild):
                        return "```\nThe current server's privilege level is already at the highest available level. However, you may still purchase this item for other servers.```"
                    return await send_with_react(channel, f"```callback-fun-shop-{user.id}_{item}-\nYou are about to upgrade the server's privilege level from 0 to 1.\nThis is irreversible. Please choose wisely.```", reacts="✅")
                raise NotImplementedError("Target item has not yet been implemented.")
        raise ValueError(f"Insufficient funds. Use {bot.get_prefix(guild)}shop for product list and cost.")

    async def _callback_(self, bot, message, reaction, user, vals, **void):
        if reaction is None or reaction.decode("utf-8", "replace") != "✅":
            return
        u_id, item = vals.split("_", 1)
        u_id = int(u_id)
        if u_id != user.id:
            return
        guild = message.guild
        try:
            product = self.products[item]
        except KeyError:
            raise LookupError(f"Sorry, we don't sell {argv} here...")
        data = bot.data.users.get(user.id, {})
        gold = data.get("gold", 0)
        diamonds = data.get("diamonds", 0)
        if len(product.cost) < 2 or diamonds >= product.cost[0]:
            if gold >= product.cost[-1]:
                if product.name == "Upgrade Server":
                    if bot.is_trusted(guild):
                        return "```\nThe current server's privilege level is already at the highest available level. However, you may still purchase this item for other servers."
                    bot.data.users.add_diamonds(-product.cost[0])
                    bot.data.users.add_gold(-product.cost[-1])
                    bot.data.trusted[guild.id] = True
                    bot.data.trusted.update()
                    return f"```{sqr_md(guild)} has been successfully elevated from 0 to 1 privilege level.```"
                raise NotImplementedError("Target item has not yet been implemented.")
        raise ValueError(f"Insufficient funds. Use {bot.get_prefix(guild)}shop for product list and cost.")


class MimicConfig(Command):
    name = ["PluralConfig", "RPConfig"]
    min_level = 0
    description = "Modifies an existing webhook mimic's attributes."
    usage = (
        "<0:mimic_id> <1:option(prefix)(name[])(avatar)"
        + "(description)(gender)(birthday)> <2:new>"
    )
    no_parse = True
    rate_limit = 1

    async def __call__(self, bot, user, message, perm, flags, args, **void):
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        m_id = "&" + str(verify_id(args.pop(0)))
        if m_id not in mimicdb:
            raise LookupError("Target mimic ID not found.")
        # Users are not allowed to modify mimics they do not control
        if not isnan(perm):
            mimics = set_dict(mimicdb, user.id, {})
            found = 0
            for prefix in mimics:
                found += mimics[prefix].count(m_id)
            if not found:
                raise PermissionError("Target mimic does not belong to you.")
        else:
            mimics = mimicdb[mimicdb[m_id].u_id]
            found = True
        mimic = mimicdb[m_id]
        opt = args.pop(0).casefold()
        args.extend(best_url(a) for a in message.attachments)
        if args:
            new = " ".join(args)
        else:
            new = None
        if opt in ("name", "username", "nickname", ""):
            setting = "name"
        elif opt in ("avatar", "icon", "url", "pfp", "image"):
            setting = "url"
        elif opt in ("status", "description"):
            setting = "description"
        elif opt in ("gender", "birthday", "prefix"):
            setting = opt
        elif opt in ("auto", "copy", "user", "auto"):
            setting = "auto"
        else:
            raise TypeError("Invalid target attribute.")
        if new is None:
            return ini_md(f"Current {setting} for {sqr_md(mimic.name)}: {sqr_md(mimic[setting])}.")
        if setting == "birthday":
            new = utc_ts(tzparse(new))
        # This limit is actually to comply with webhook usernames
        elif setting == "name":
            if len(new) > 80:
                raise OverflowError("Name must be 80 or fewer in length.")
        # Prefixes must not be too long
        elif setting == "prefix":
            if len(new) > 16:
                raise OverflowError("Prefix must be 16 or fewer in length.")
            for prefix in mimics:
                with suppress(ValueError, IndexError):
                    mimics[prefix].remove(m_id)
            if new in mimics:
                mimics[new].append(m_id)
            else:
                mimics[new] = [m_id]
        elif setting == "url":
            urls = await bot.follow_url(new, best=True)
            new = urls[0]
        # May assign a user to the mimic
        elif setting == "auto":
            if new.casefold() in ("none", "null", "0", "false", "f"):
                new = None
            else:
                mim = None
                try:
                    mim = verify_id(new)
                    user = await bot.fetch_user(mim)
                    if user is None:
                        raise EOFError
                    new = user.id
                except:
                    try:
                        mimi = bot.get_mimic(mim, user)
                        new = mimi.id
                    except:
                        raise LookupError("Target user or mimic ID not found.")
        elif setting != "description":
            if len(new) > 512:
                raise OverflowError("Must be 512 or fewer in length.")
        name = mimic.name
        mimic[setting] = new
        update()
        return css_md(f"Changed {setting} for {sqr_md(name)} to {sqr_md(new)}.")


class Mimic(Command):
    name = ["RolePlay", "Plural", "RP", "RPCreate"]
    min_level = 0
    description = "Spawns a webhook mimic with an optional username and icon URL, or lists all mimics with their respective prefixes."
    usage = "<0:prefix> <1:user[]> <1:name[]> <2:url[]> <disable(?d)> <debug(?z)>"
    flags = "aedzf"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = (1, 2)
    
    async def __call__(self, bot, message, user, perm, flags, args, argv, **void):
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        args.extend(best_url(a) for a in reversed(message.attachments))
        if len(args) == 1 and "d" not in flags:
            user = await bot.fetch_user(verify_id(argv))
        mimics = set_dict(mimicdb, user.id, {})
        if not argv or (len(args) == 1 and "d" not in flags):
            if "d" in flags:
                # This deletes all mimics for the current user
                if "f" not in flags and len(mimics) > 1:
                    return css_md(sqr_md(f"WARNING: {len(mimics)} MIMICS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."))
                mimicdb.pop(user.id)
                update()
                return italics(css_md(f"Successfully removed all {sqr_md(len(mimics))} webhook mimics for {sqr_md(user)}."))
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-fun-mimic-"
                + str(user.id) + "_0"
                + "-\nLoading Mimic database...```*"
            )
        u_id = user.id
        prefix = args.pop(0)
        if "d" in flags:
            try:
                mlist = mimics[prefix]
                if mlist is None:
                    raise KeyError
                if len(mlist):
                    m_id = mlist.pop(0)
                    mimic = mimicdb.pop(m_id)
                else:
                    mimics.pop(prefix)
                    update()
                    raise KeyError
                if not mlist:
                    mimics.pop(prefix)
            except KeyError:
                mimic = bot.get_mimic(prefix, user)
                # Users are not allowed to delete mimics that do not belong to them
                if not isnan(perm) and mimic.u_id != user.id:
                    raise PermissionError("Target mimic does not belong to you.")
                mimics = mimicdb[mimic.u_id]
                user = await bot.fetch_user(mimic.u_id)
                m_id = mimic.id
                for prefix in mimics:
                    with suppress(ValueError, IndexError):
                        mimics[prefix].remove(m_id)
                mimicdb.pop(mimic.id)
            update()
            return italics(css_md(f"Successfully removed webhook mimic {sqr_md(mimic.name)} for {sqr_md(user)}."))
        if not prefix:
            raise IndexError("Prefix must not be empty.")
        if len(prefix) > 16:
            raise OverflowError("Prefix must be 16 or fewer in length.")
        if " " in prefix:
            raise TypeError("Prefix must not contain spaces.")
        # This limit is ridiculous. I like it.
        if sum(len(i) for i in iter(mimics.values())) >= 32768:
            raise OverflowError(f"Mimic list for {user} has reached the maximum of 32768 items. Please remove an item to add another.")
        dop = None
        utcn = utc_dt()
        mid = discord.utils.time_snowflake(utcn)
        ctime = utc()
        m_id = "&" + str(mid)
        mimic = None
        # Attempt to create a new mimic, a mimic from a user, or a copy of an existing mimic.
        if len(args):
            if len(args) > 1:
                urls = await bot.follow_url(args[-1], best=True)
                url = urls[0]
                name = " ".join(args[:-1])
            else:
                mim = 0
                try:
                    mim = verify_id(args[-1])
                    user = await bot.fetch_user(mim)
                    if user is None:
                        raise EOFError
                    dop = user.id
                    name = user.name
                    url = best_url(user)
                except:
                    try:
                        mimi = bot.get_mimic(mim, user)
                        dop = mimi.id
                        mimic = copy.deepcopy(mimi)
                        mimic.id = m_id
                        mimic.u_id = u_id
                        mimic.prefix = prefix
                        mimic.count = mimic.total = 0
                        mimic.created_at = ctime
                        mimic.auto = dop
                    except:
                        name = args[0]
                        url = "https://cdn.discordapp.com/embed/avatars/0.png"
        else:
            name = user.name
            url = best_url(user)
        # This limit is actually to comply with webhook usernames
        if len(name) > 80:
            raise OverflowError("Name must be 80 or fewer in length.")
        while m_id in mimics:
            mid += 1
            m_id = "&" + str(mid)
        if mimic is None:
            mimic = cdict(
                id=m_id,
                u_id=u_id,
                prefix=prefix,
                auto=dop,
                name=name,
                url=url,
                description="",
                gender="N/A",
                birthday=ctime,
                created_at=ctime,
                count=0,
                total=0,
            )
        mimicdb[m_id] = mimic
        if prefix in mimics:
            mimics[prefix].append(m_id)
        else:
            mimics[prefix] = [m_id]
        update()
        out = f"Successfully added webhook mimic {sqr_md(mimic.name)} with prefix {sqr_md(mimic.prefix)} and ID {sqr_md(mimic.id)}"
        if dop is not None:
            out += f", bound to user [{user_mention(dop) if type(dop) is int else f'<{dop}>'}]"
        return css_md(out)

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_", 1)]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm <= inf:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        user = await bot.fetch_user(u_id)
        mimics = mimicdb.get(user.id, {})
        for k in tuple(mimics):
            if not mimics[k]:
                mimics.pop(k)
                update()
        page = 24
        last = max(0, len(mimics) - page)
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
            "callback-fun-mimic-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not mimics:
            content += f"No currently enabled webhook mimics for {str(user).replace('`', '')}.```*"
            msg = ""
        else:
            content += f"{len(mimics)} currently enabled webhook mimics for {str(user).replace('`', '')}:```*"
            key = lambda x: lim_str("⟨" + ", ".join(i + ": " + (str(no_md(mimicdb[i].name)), "[<@" + str(getattr(mimicdb[i], "auto", "None")) + ">]")[bool(getattr(mimicdb[i], "auto", None))] for i in iter(x)) + "⟩", 1900 / len(mimics))
            msg = ini_md(iter2str({k: mimics[k] for k in sorted(mimics)[pos:pos + page]}, key=key))
        emb = discord.Embed(
            description=content + msg,
            colour=rand_colour(),
        )
        emb.set_author(**get_author(user))
        more = len(mimics) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(react.decode("utf-8")))
                await asyncio.sleep(0.5)


class MimicSend(Command):
    name = ["RPSend", "PluralSend"]
    min_level = 0
    description = "Sends a message using a webhook mimic, to the target channel."
    usage = "<0:mimic> <1:channel> <2:string>"
    no_parse = True
    rate_limit = 0.5

    async def __call__(self, bot, channel, message, user, perm, args, **void):
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        mimics = set_dict(mimicdb, user.id, {})
        prefix = args.pop(0)
        c_id = verify_id(args.pop(0))
        channel = await bot.fetch_channel(c_id)
        guild = channel.guild
        msg = " ".join(args)
        if not msg:
            raise IndexError("Message is empty.")
        perm = bot.get_perms(user.id, guild)
        try:
            mlist = mimics[prefix]
            if mlist is None:
                raise KeyError
            m = [bot.get_mimic(verify_id(p)) for p in mlist]
        except KeyError:
            mimic = bot.get_mimic(verify_id(prefix))
            m = [mimic]
        admin = not inf > perm
        try:
            enabled = bot.data.enabled[channel.id]
        except KeyError:
            enabled = ()
        # Because this command operates across channels and servers, we need to make sure these cannot be sent to channels without this command enabled
        if not admin and "fun" not in enabled:
            raise PermissionError("Not permitted to send into target channel.")
        if m:
            msg = escape_everyone(msg)
            if msg.startswith("/tts "):
                msg = msg[5:]
                tts = True
            else:
                tts = False
            for mimic in m:
                await bot.data.mimics.updateMimic(mimic, guild)
                name = mimic.name
                url = mimic.url
                await wait_on_none(bot.send_as_webhook(channel, msg, username=name, avatar_url=url, tts=tts))
                mimic.count += 1
                mimic.total += len(msg)
            create_task(message.add_reaction("👀"))


class UpdateMimics(Database):
    name = "mimics"
    user = True

    async def _nocommand_(self, message, **void):
        if not message.content:
            return
        user = message.author
        if user.id in self.data:
            bot = self.bot
            perm = bot.get_perms(user.id, message.guild)
            if perm < 0:
                return
            admin = not inf > perm
            if message.guild is not None:
                try:
                    enabled = bot.data.enabled[message.channel.id]
                except KeyError:
                    enabled = ()
            else:
                enabled = list(bot.categories)
            # User must have permission to use ~mimicsend in order to invoke by prefix
            if admin or "fun" in enabled:
                database = self.data[user.id]
                msg = message.content
                async with ExceptionSender(message.channel, Exception):
                    # Stack multiple messages to send, may be separated by newlines
                    sending = alist()
                    channel = message.channel
                    for line in msg.splitlines():
                        found = False
                        # O(1) time complexity per line regardless of how many mimics a user is assigned
                        if len(line) > 2 and " " in line:
                            i = line.index(" ")
                            prefix = line[:i]
                            if prefix in database:
                                mimics = database[prefix]
                                if mimics:
                                    line = line[i + 1:].strip(" ")
                                    for m in mimics:
                                        sending.append(cdict(m_id=m, msg=line))
                                    found = True
                        if not sending:
                            break
                        if not found:
                            sending[-1].msg += "\n" + line
                    if sending:
                        create_task(bot.silent_delete(message))
                        for k in sending:
                            mimic = self.data[k.m_id]
                            await self.updateMimic(mimic, guild=message.guild)
                            name = mimic.name
                            url = mimic.url
                            msg = escape_everyone(k.msg)
                            if msg.startswith("/tts "):
                                msg = msg[5:]
                                tts = True
                            else:
                                tts = False
                            await wait_on_none(bot.send_as_webhook(channel, msg, username=name, avatar_url=url, tts=tts))
                            mimic.count += 1
                            mimic.total += len(k.msg)
                            bot.data.users.add_xp(user, math.sqrt(len(msg)) * 2)

    async def updateMimic(self, mimic, guild=None, it=None):
        if set_dict(mimic, "auto", None):
            bot = self.bot
            mim = 0
            try:
                mim = verify_id(mimic.auto)
                if guild is not None:
                    user = guild.get_member(mim)
                if user is None:
                    user = await bot.fetch_user(mim)
                if user is None:
                    raise LookupError
                mimic.name = user.display_name
                mimic.url = best_url(user)
            except (discord.NotFound, LookupError):
                try:
                    mimi = bot.get_mimic(mim)
                    if it is None:
                        it = {}
                    # If we find the same mimic twice, there is an infinite loop
                    elif mim in it:
                        raise RecursionError("Infinite recursive loop detected.")
                    it[mim] = True
                    if not len(it) & 255:
                        await asyncio.sleep(0.2)
                    await self.updateMimic(mimi, guild=guild, it=it)
                    mimic.name = mimi.name
                    mimic.url = mimi.url
                except LookupError:
                    mimic.name = str(mimic.auto)
                    mimic.url = "https://cdn.discordapp.com/embed/avatars/0.png"

    async def __call__(self):
        with tracebacksuppressor(SemaphoreOverflowError):
            async with self._semaphore:
                async with delay(2):
                    # Garbage collector for unassigned mimics
                    i = 1
                    for m_id in tuple(self.data):
                        if type(m_id) is str:
                            mimic = self.data[m_id]
                            try:
                                if mimic.u_id not in self.data or mimic.id not in self.data[mimic.u_id][mimic.prefix]:
                                    self.data.pop(m_id)
                                    self.update()
                            except:
                                self.data.pop(m_id)
                                self.update()
                        if not i % 8191:
                            await asyncio.sleep(0.45)
                        i += 1