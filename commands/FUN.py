try:
    from common import *
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath('..'))
    os.chdir("..")
    from common import *

import nekos
print = PRINT


try:
    alexflipnote_key = AUTH["alexflipnote_key"]
    if not alexflipnote_key:
        raise
except:
    alexflipnote_key = None
    print("WARNING: alexflipnote_key not found. Unable to use API to generate images.")
try:
    giphy_key = AUTH["giphy_key"]
    if not giphy_key:
        raise
except:
    giphy_key = None
    print("WARNING: giphy_key not found. Unable to use API to search images.")


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
    name = ["2048", "🎮"]
    description = "Plays a game of 2048 using reactions. Gained points are rewarded as gold."
    usage = "<0:dimension_sizes(4x4)>* <1:dimension_count(2)>? <public{?p}|special_tiles{?s}|insanity_mode{?i}|easy_mode{?e}>*"
    flags = "pies"
    rate_limit = (3, 9)
    reacts = ("⬅️", "➡️", "⬆️", "⬇️", "⏪", "⏩", "⏫", "⏬", "◀️", "▶️", "🔼", "🔽", "👈", "👉", "👆", "👇")
    directions = demap((r.encode("utf-8"), i) for i, r in enumerate(reacts))
    directions[b'\xf0\x9f\x92\xa0'] = -2
    directions[b'\xe2\x86\xa9\xef\xb8\x8f'] = -1
    slash = ("2048",)

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
                    create_task(message.add_reaction(as_str(react)))
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
                    await asyncio.sleep(0.2)
                return
        if data is not None:
            # Update message if gamestate has been changed
            if u_id == 0:
                u = None
            elif user.id == u_id:
                u = user
            else:
                u = bot.get_user(u_id, replace=True)
            colour = await bot.get_colour(u)
            emb = discord.Embed(colour=colour)
            if u is None:
                emb.set_author(name="@everyone", icon_url=bot.discord_icon)
            else:
                emb.set_author(**get_author(u))
            content = "*```callback-fun-text2048-" + str(u_id) + "_" + str(mode) + "-" + "_".join(str(i) for i in size) + "-" + as_str(data) + "\nPlaying 2048...```*"
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
                size = await recursive_coro([bot.eval_math(i) for i in argv.split("x")])
            else:
                if len(args) > 1:
                    dims = args.pop(-1)
                    dims = await bot.eval_math(dims)
                else:
                    dims = 2
                if dims <= 0:
                    raise ValueError("Invalid amount of dimensions specified.")
                width = await bot.eval_math(" ".join(args))
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
    description = "Plays a slot machine game. Costs gold to play, can yield gold and diamonds."
    usage = "<bet{50}>? <skip_animation{?s}>?"
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
    slash = ("Slots",)

    def select(self):
        x = random.random()
        if x < 1 / 32:
            return "💎"
        elif x < 3 / 32:
            return "❤️"
        return choice("🍒🍎🍇🍋🍉🍌")

    def generate(self, rate="low", count=3):
        x = random.random()
        if rate == "high":
            if x < 1 / 5:
                count = 3
            elif x < 8 / 15:
                count = 2
            else:
                count = 1
        else:
            if x < 1 / 7:
                count = 3
            elif x < 3 / 7:
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
        if argv:
            bet = await self.bot.eval_math(argv)
            if bet < b1:
                raise ValueError(f"Minimum bet is {b1} coins.")
        else:
            bet = b1
        if not bet <= self.bot.data.users.get(user.id, {}).get("gold", 0):
            raise OverflowError("Bet cannot be greater than your balance.")
        self.bot.data.users.add_gold(user, -bet)
        skip = int("s" in flags)
        return f"*```callback-fun-slotmachine-{user.id}_{bet}_{skip}-\nLoading Slot Machine...```*"

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        spl = [int(i) for i in vals.split("_", 2)]
        if len(spl) < 3:
            spl.append(0)
        u_id, bet, skip = spl
        if reaction is None or as_str(reaction) == "⤵️":
            if reaction is None:
                create_task(message.add_reaction("⤵️"))
                user = await bot.fetch_user(u_id)
            else:
                if bet > bot.data.users.get(user.id, {}).get("gold", 0):
                    raise OverflowError("Bet cannot be greater than your balance.")
                bot.data.users.add_gold(user, -bet)
            wheel_true = self.generate("high" if bet <= 50 else "low")
            wheel_display = [None] * 3 if not skip else wheel_true
            wheel_order = deque(shuffle(range(3))) if not skip else deque((0, ))
            colour = await bot.get_colour(user)
            emb = discord.Embed(colour=colour).set_author(**get_author(user))
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
    description = "Pays a specified amount of coins to the target user."
    usage = "<0:user> <1:amount(1)>?"
    rate_limit = 0.5

    async def __call__(self, bot, user, args, guild, **void):
        if not args:
            raise ArgumentError("Please input target user.")
        target = await bot.fetch_user_member(args.pop(0), guild)
        if target.id == bot.id:
            return "\u200bI appreciate the generosity, but I have enough already :3"
        if args:
            amount = await bot.eval_math(" ".join(args))
        else:
            amount = 1
        if amount <= 0:
            raise ValueError("Please input a valid amount of coins.")
        if not amount <= bot.data.users.get(user.id, {}).get("gold", 0):
            raise OverflowError("Payment cannot be greater than your balance.")
        bot.data.users.add_gold(user, -amount)
        bot.data.users.add_gold(target, amount)
        if user.id != target.id:
            bot.data.dailies.progress_quests(user, "pay", amount)
        return css_md(f"{sqr_md(user)} has paid {sqr_md(amount)} coins to {sqr_md(target)}.")


class React(Command):
    server_only = True
    name = ["AutoReact"]
    min_level = 2
    description = "Causes ⟨MIZA⟩ to automatically assign a reaction to messages containing the substring."
    usage = "<0:react_to>? <1:react_data>? <disable{?d}>?"
    flags = "aedzf"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = (1, 2)
    slash = True

    async def __call__(self, bot, flags, guild, message, user, argv, args, **void):
        update = self.data.reacts.update
        following = bot.data.reacts
        curr = set_dict(following, guild.id, mdict())
        if type(curr) is not mdict:
            following[guild.id] = curr = mdict(curr)
        if not argv:
            if "d" in flags:
                # This deletes all auto reacts for the current guild
                if "f" not in flags and len(curr) > 1:
                    return css_md(sqr_md(f"WARNING: {len(curr)} ITEMS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), force=True)
                if guild.id in following:
                    following.pop(guild.id)
                return italics(css_md(f"Successfully removed all {sqr_md(len(curr))} auto reacts for {sqr_md(guild)}."))
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-fun-react-"
                + str(user.id) + "_0"
                + "-\nLoading React database...```*"
            )
        if "d" in flags:
            a = full_prune(args[0])
            if a in curr:
                curr.pop(a)
                update(guild.id)
                return italics(css_md(f"Removed {sqr_md(a)} from the auto react list for {sqr_md(guild)}."))
            else:
                raise LookupError(f"{a} is not in the auto react list.")
        lim = 64 << bot.is_trusted(guild.id) * 2 + 1
        if curr.count() >= lim:
            raise OverflowError(f"React list for {guild} has reached the maximum of {lim} items. Please remove an item to add another.")
        # Limit substring length to 64
        a = unicode_prune(" ".join(args[:-1])).casefold()[:64]
        try:
            e_id = int(args[-1])
        except:
            emoji = args[-1]
        else:
            emoji = await bot.fetch_emoji(e_id)
        emoji = str(emoji)
        # This reaction indicates that the emoji was valid
        await message.add_reaction(emoji)
        curr.append(a, emoji)
        following[guild.id] = mdict({i: curr[i] for i in sorted(curr)})
        return css_md(f"Added {sqr_md(a)} ➡️ {sqr_md(emoji)} to the auto react list for {sqr_md(guild)}.")

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_", 1)]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        following = bot.data.reacts
        curr = following.get(guild.id, mdict())
        page = 16
        last = max(0, len(curr) - page)
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
            "callback-fun-react-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not curr:
            content += f"No currently assigned auto reactions for {str(guild).replace('`', '')}.```*"
            msg = ""
        else:
            content += f"{len(curr)} auto reactions currently assigned for {str(guild).replace('`', '')}:```*"
            key = lambda x: "\n" + ", ".join(x)
            msg = ini_md(iter2str({k: curr[k] for k in tuple(curr)[pos:pos + page]}, key=key))
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(as_str(react)))
                await asyncio.sleep(0.2)


class UpdateReacts(Database):
    name = "reacts"

    async def _nocommand_(self, text, edit, orig, message, **void):
        if message.guild is None or not orig:
            return
        g_id = message.guild.id
        data = self.data
        if g_id in data:
            with tracebacksuppressor(ZeroDivisionError):
                following = self.data[g_id]
                if type(following) != mdict:
                    following = self.data[g_id] = mdict(following)
                reacting = {}
                for k in following:
                    if is_alphanumeric(k) and " " not in k:
                        words = text.split()
                    else:
                        words = full_prune(message.content)
                    if k in words:
                        emojis = following[k]
                        # Store position for each keyword found
                        reacting[words.index(k) / len(words)] = emojis
                # Reactions sorted by their order of appearance in the message
                for r in sorted(reacting):
                    for react in reacting[r]:
                        create_task(self.add_reaction_conditional(message, react, emojis, g_id))
                        await asyncio.sleep(0.2)

    async def add_reaction_conditional(self, message, react, emojis, g_id):
        try:
            await message.add_reaction(react)
        except discord.HTTPException as ex:
            if "10014" in repr(ex):
                emojis.remove(react)
                self.update(g_id)


class EmojiList(Command):
    description = "Sets a custom alias for an emoji, usable by ~autoemoji."
    usage = "(add|delete)? <name> <id>"
    flags = "aed"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']

    async def __call__(self, bot, flags, user, name, argv, args, **void):
        data = bot.data.emojilists
        if "d" in flags:
            try:
                e_id = bot.data.emojilists[user.id].pop(args[0])
            except KeyError:
                raise KeyError(f'Emoji name "{args[0]}" not found.')
            return italics(css_md(f"Successfully removed emoji alias {sqr_md(args[0])}: {sqr_md(e_id)} for {sqr_md(user)}."))
        elif argv:
            try:
                name, e_id = argv.rsplit(None, 1)
            except ValueError:
                raise ArgumentError("Please input alias followed by emoji, separated by a space.")
            name = name.strip(":")
            if not regexp("[A-Za-z0-9\\-~_]{1,32}").fullmatch(name):
                raise ArgumentError("Emoji aliases may only contain 1~32 alphanumeric characters, dashes, tildes and underscores.")
            e_id = e_id.rsplit(":", 1)[-1].rstrip(">")
            if not e_id.isnumeric():
                raise ArgumentError("Only custom emojis are supported.")
            e_id = int(e_id)
            animated = await create_future(bot.is_animated, e_id, verify=True)
            if animated is None:
                raise LookupError(f"Emoji {e_id} does not exist.")
            bot.data.emojilists.setdefault(user.id, {})[name] = e_id
            bot.data.emojilists.update(user.id)
            return ini_md(f"Successfully added emoji alias {sqr_md(name)}: {sqr_md(e_id)} for {sqr_md(user)}.")
        return (
            "*```" + "\n" * ("z" in flags) + "callback-fun-emojilist-"
            + str(user.id) + "_0"
            + "-\nLoading EmojiList database...```*"
        )
    
    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_", 1)]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm <= inf:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        following = bot.data.emojilists
        curr = {}
        for k, v in sorted(following.get(user.id, {}).items(), key=lambda n: full_prune(n[0])):
            try:
                me = await bot.min_emoji(v)
            except LookupError:
                following[user.id].pop(k)
                continue
            curr[f":{k}:"] = f"({v})` {me}"
        page = 16
        last = max(0, len(curr) - page)
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
            "callback-fun-emojilist-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not curr:
            content += f"No currently assigned emoji aliases for {str(user).replace('`', '')}.```*"
            msg = ""
        else:
            content += f"{len(curr)} emoji aliases currently assigned for {str(user).replace('`', '')}:```*"
            key = lambda x: "\n" + ", ".join(x)
            msg = iter2str({k + " " * (32 - len(k)): curr[k] for k in tuple(curr)[pos:pos + page]}, left="`", right="")
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(as_str(react)))
                await asyncio.sleep(0.2)


class UpdateEmojiLists(Database):
    name = "emojilists"
    user = True


class Dogpile(Command):
    server_only = True
    min_level = 2
    description = "Causes ⟨MIZA⟩ to automatically imitate users when 3+ of the same messages are posted in a row. Grants XP and gold when triggered."
    usage = "(enable|disable)?"
    flags = "aed"
    rate_limit = 0.5

    async def __call__(self, flags, guild, name, **void):
        update = self.data.dogpiles.update
        bot = self.bot
        following = bot.data.dogpiles
        curr = following.get(guild.id, False)
        if "d" in flags:
            if guild.id in following:
                following.pop(guild.id)
            return css_md(f"Disabled dogpile imitating for {sqr_md(guild)}.")
        if "e" in flags or "a" in flags:
            following[guild.id] = True
            return css_md(f"Enabled dogpile imitating for {sqr_md(guild)}.")
        if curr:
            return ini_md(f"Dogpile imitating is currently enabled in {sqr_md(guild)}.")
        return ini_md(f'Dogpile imitating is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


class UpdateDogpiles(Database):
    name = "dogpiles"

    async def _nocommand_(self, edit, message, **void):
        if message.guild is None or not message.content:
            return
        g_id = message.guild.id
        following = self.data
        dogpile = following.get(g_id)
        if dogpile:
            u_id = message.author.id
            c_id = message.channel.id
            if not edit:
                content = zwremove(message.content)
                if not content:
                    return
                try:
                    number = round_min(content)
                except ValueError:
                    if len(content) == 1:
                        last_number = number = content
                        add = None
                    else:
                        number = None
                else:
                    numbers = deque((number,))
                mcount = 0
                count = 0
                last_author_id = u_id
                stopped = False
                async for m in self.bot.history(message.channel, limit=100):
                    if m.id == message.id:
                        continue
                    c = zwremove(m.content)
                    if not c:
                        break
                    if number is not None:
                        if type(number) is str:
                            if len(c) != 1:
                                break
                            n = ord(c)
                            if add is None:
                                add = n - ord(number)
                            elif n - add != ord(number):
                                break
                            number = c
                        else:
                            try:
                                n = round_min(c)
                            except:
                                break
                            numbers.appendleft(n)
                    elif c != content:
                        break
                    if m.author.id == last_author_id:
                        break
                    if m.author.id == self.bot.id:
                        stopped = True
                    elif not stopped:
                        count += 1
                    mcount += 1
                    if mcount >= 11:
                        break
                    last_author_id = m.author.id
                # print(content, count)
                if count >= 2 and random.random() >= 2 / (count + 0.5):
                    if number is not None:
                        if type(number) is str:
                            content = chr(ord(last_number) - add)
                        else:
                            n = await create_future(predict_next, numbers)
                            if not n:
                                return
                            content = str(n)
                        content = content.strip()
                        if not content:
                            return
                    print(message.channel, content, mcount)
                    if content[0].isascii() and content[:2] != "<:":
                        content = lim_str("\u200b" + content, 2000)
                    create_task(message.channel.send(content, tts=message.tts))
                    self.bot.data.users.add_xp(message.author, len(message.content) / 2 + 16)
                    self.bot.data.users.add_gold(message.author, len(message.content) / 4 + 32)


class Daily(Command):
    name = ["Quests", "Quest", "Tasks", "Challenges", "Dailies"]
    description = "Shows your list of daily quests."
    rate_limit = 1

    async def __call__(self, bot, user, channel, **void):
        data = bot.data.dailies.get(user)
        colour = await bot.get_colour(user)
        emb = discord.Embed(title="Daily Quests", colour=colour).set_author(**get_author(user))
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
        if as_str(reaction) != "✅":
            return
        u_id = vals
        if str(user.id) != u_id:
            return
        data = bot.data.dailies.collect(user)
        colour = await bot.get_colour(user)
        emb = discord.Embed(title="Daily Quests", colour=colour).set_author(**get_author(user))
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
    no_delete = True

    def __load__(self, **void):
        self.typing = {}
        self.generator = alist()
        self.initialize()

    def get(self, user):
        data = self.data.get(user.id)
        if data is None or utc() - data["time"] >= 86400:
            data = self.data[user.id] = dict(quests=self.generate(user), time=zerot())
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
            self.update(user.id)
        return data
    
    def add_quest(self, weight, action, name, x=1, gold=0, diamonds=0, **kwargs):
        if weight > 0:

            def func(action, name, level, x, gold, diamonds, **kwargs):
                if callable(x):
                    x = x(level)
                x = round_random(x)
                if callable(gold):
                    gold = gold(x)
                if callable(diamonds):
                    diamonds = diamonds(x)
                if callable(name):
                    name = name(x)
                elif "{x}" in name:
                    name = name.format(x=x)
                quest = cdict(
                    action=action,
                    name=name,
                    progress=0,
                    required=x,
                    gold=round_random(gold),
                    diamonds=round_random(diamonds),
                )
                quest.update(kwargs)
                return quest

            self.generator.extend(repeat(lambda level: func(action, name, level, x, gold, diamonds, **kwargs), weight))

    def initialize(self):
        add_quest = self.add_quest
        scale = lambda low, high, level_bonus=0, multiple=0: lambda level: round_random_multiple(random.randint(low, high) + level * level_bonus * random.random(), multiple)
        add_quest(100, "send", "Post {x} messages", gold=lambda x: x << 1,
            x=scale(70, 180, 3, 3))
        add_quest(70, "music", lambda x: f"Listen to {sec2time(x)} of my music", gold=nofunc, diamonds=lambda x: xrand(1, x) / 60,
            x=scale(180, 480, 13, 60))
        for catg in "main string voice image fun".split():
            add_quest(13, "category", "Use {x} " + catg + " commands", catg=catg, gold=lambda x: x * 9,
                x=scale(9, 17, 0.3))
        add_quest(60, "typing", lambda x: f"Type for {sec2time(x)}", gold=lambda x: x * 2.5,
            x=scale(150, 400, 11, 30))
        add_quest(52, "first", lambda x: f"Be the first to post in a channel since {sec2time(x)} ago", gold=lambda x: x >> 3, diamonds=1,
            x=scale(1800, 3600, 240, 300))
        add_quest(50, "reply", "Reply to {x} messages", gold=lambda x: x * 12,
            x=scale(23, 40, 1.1))
        add_quest(50, "channel", "Send messages in {x} different channels", gold=lambda x: x * 24,
            x=scale(5, 8, 0.06))
        add_quest(45, "react", "Add {x} reactions", gold=lambda x: x * 5,
            x=scale(20, 39, 1, 2))
        add_quest(45, "reacted", "Have {x} reactions added to your messages by other users", gold=lambda x: x * 15, diamonds=lambda x: xrand(1, x) / 8,
            x=scale(15, 31, 0.75, 2))
        add_quest(42, "xp", "Earn {x} experience", gold=lambda x: x >> 2, diamonds=1,
            x=scale(1000, 2000, 80, 50))
        add_quest(40, "word", "Send {x} total words of text", gold=lambda x: x / sqrt(5),
            x=scale(400, 600, 15, 5))
        add_quest(40, "text", "Send {x} total characters of text", gold=lambda x: x >> 3,
            x=scale(1600, 2400, 60, 10))
        add_quest(37, "url", "Send {x} attachments or links", gold=lambda x: x * 16,
            x=scale(12, 25, 0.15))
        add_quest(35, "command", "Use {x} commands", gold=lambda x: x * 9,
            x=scale(24, 40, 0.4))
        add_quest(30, "talk", "Talk to me {x} times", gold=lambda x: x * 7, diamonds=xrand(2, 4),
            x=lambda level: xrand(10, 21))
        add_quest(20, "pay", "Pay {x} to other users", gold=lambda x: x >> 1, diamonds=lambda x: xrand(0, 2),
            x=scale(400, 900, 70, 100))
        add_quest(18, "diamond", "Earn 1 diamond", required=1, gold=lambda x: x * 50, diamonds=1,
            x=nofunc)
        add_quest(15, "invite", "Invite me to a server and/or react to the join message", required=1, diamonds=lambda x: 50 + x * 2 / 3,
            x=nofunc)

    def generate(self, user):
        if user.id == self.bot.id or self.bot.is_blacklisted(user.id):
            return dict(quests=(), time=inf)
        level = self.bot.data.users.xp_to_level(self.bot.data.users.get_xp(user))
        quests = alist()
        req = min(20, level + 5 >> 1)
        att = 0
        while len(quests) < req and att < req << 1:
            q = choice(self.generator)(level)
            if not quests or q.action not in (e.action for e in quests[-5:]):
                q.progress = 0
                quests.append(q)
            att += 1
        return quests.appendleft(cdict(action=None, name="Daily rewards", progress=1, required=1, gold=level * 30 + 400))

    def progress_quests(self, user, action, value=1):
        if user.id == self.bot.id or self.bot.is_blacklisted(user.id):
            return
        data = self.get(user)
        quests = data["quests"]
        for i in range(min(5, len(quests))):
            quest = quests[i]
            if quest.action == action:
                if callable(value):
                    value(quest)
                else:
                    quest.progress += value
                self.update(user.id)

    async def valid_message(self, message):
        user = message.author
        self.progress_quests(user, "send")
        self.progress_quests(user, "text", get_message_length(message))
        self.progress_quests(user, "word", get_message_words(message))
        self.progress_quests(user, "url", len(message.attachments) + len(message.embeds) + len(find_urls(message.content)))
        if getattr(message, "reference", None):
            self.progress_quests(user, "reply")
        
        def progress_channel(quest):
            channels = quest.setdefault("channels", set())
            channels.add(message.channel.id)
            quest.progress = len(channels)

        self.progress_quests(user, "channel", progress_channel)
        if user.id in self.typing:
            t = utc()
            self.progress_quests(user, "typing", t - self.typing.pop(user.id, t))
        last = await self.bot.get_last_message(message.channel, key=lambda m: m.id < message.id)
        since = id2td(message.id - last.id)
        self.progress_quests(user, "first", lambda quest: quest.__setitem__("progress", max(quest.progress, since)))

    def _command_(self, user, command, loop=False, **void):
        if not loop:
            self.progress_quests(user, "command")

            def progress_category(quest):
                if quest.catg == command.catg.casefold():
                    quest.progress += 1

            self.progress_quests(user, "category", progress_category)

    def _typing_(self, user, **void):
        if user.id not in self.data:
            self.typing.pop(user.id, None)
            return
        if user.id in self.typing:
            if utc() - self.typing[user.id] > 10:
                self.progress_quests(user, "typing", 10)
                self.typing[user.id] = utc()
        else:
            self.typing[user.id] = utc()

    def _reaction_add_(self, message, user, **void):
        self.progress_quests(user, "react")
        if message.author.id != user.id and user.id != self.bot.id:
            self.progress_quests(message.author, "reacted")


class Wallet(Command):
    name = ["Level", "Bal", "Balance"]
    description = "Shows the target users' wallet."
    usage = "<users>*"
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
            url = await self.bot.get_proxy_url(user)
            bot.send_as_embeds(channel, description, thumbnail=url, author=get_author(user))

    join_cache = {}

    async def _callback_(self, bot, message, reaction, user, vals, **void):
        ts = int(float(vals))
        if utc() - ts > 86400:
            self.join_cache.pop(message.id, None)
            return
        if reaction is None or as_str(reaction) != "✅":
            return
        cache = set_dict(self.join_cache, message.id, set())
        if len(cache) > 256:
            cache.pop()
        if user.id in cache:
            return
        cache.add(user.id)
        bot.data.dailies.progress_quests(user, "invite")


class Shop(Command):
    description = "Displays the shop system, or purchases an item."
    usage = "<item[]>"
    rate_limit = 1

    products = cdict(
        upgradeserver=cdict(
            name="Upgrade Server",
            cost=[240, 30720],
            description="Upgrades the server's privilege level, granting access to all command categories and reducing command cooldown.",
        ),
    )

    async def __call__(self, bot, guild, channel, user, message, argv, **void):
        if not argv:
            desc = ""
            for product in self.products.values():
                cost = await bot.as_rewards(*product.cost)
                description = ini_md(f"{sqr_md(product.name)} {cost}\n{product.description}")
            return bot.send_as_embeds(channel, description, title="Shop", author=get_author(user), reference=message)
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
                    return await send_with_react(channel, f"```callback-fun-shop-{user.id}_{item}-\nYou are about to upgrade the server's privilege level from 0 to 1.\nThis is irreversible. Please choose wisely.```", reacts="✅", reference=message)
                raise NotImplementedError("Target item has not yet been implemented.")
        raise ValueError(f"Insufficient funds. Use {bot.get_prefix(guild)}shop for product list and cost.")

    async def _callback_(self, bot, message, reaction, user, vals, **void):
        if reaction is None or as_str(reaction) != "✅":
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
                    bot.data.users.add_diamonds(user, -product.cost[0])
                    bot.data.users.add_gold(user, -product.cost[-1])
                    bot.data.trusted[guild.id] = True
                    return f"```{sqr_md(guild)} has been successfully elevated from 0 to 1 privilege level.```"
                raise NotImplementedError("Target item has not yet been implemented.")
        raise ValueError(f"Insufficient funds. Use {bot.get_prefix(guild)}shop for product list and cost.")


class MimicConfig(Command):
    name = ["PluralConfig", "RPConfig"]
    description = "Modifies an existing webhook mimic's attributes."
    usage = "<0:mimic_id> (prefix|name|avatar|description|gender|birthday)? <1:new>?"
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
            u_id = user.id
            mimics = set_dict(mimicdb, u_id, {})
            found = 0
            for prefix in mimics:
                found += mimics[prefix].count(m_id)
            if not found:
                raise PermissionError("Target mimic does not belong to you.")
        else:
            u_id = mimicdb[m_id].u_id
            mimics = mimicdb[u_id]
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
        update(m_id)
        update(u_id)
        return css_md(f"Changed {setting} for {sqr_md(name)} to {sqr_md(new)}.")


class Mimic(Command):
    name = ["RolePlay", "Plural", "RP", "RPCreate"]
    description = "Spawns a webhook mimic with an optional username and icon URL, or lists all mimics with their respective prefixes. Mimics require permission level of 1 to invoke."
    usage = "<0:prefix>? <1:user|name>? <2:url[]>? <delete{?d}>?"
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
                    return css_md(sqr_md(f"WARNING: {len(mimics)} MIMICS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), force=True)
                mimicdb.pop(user.id)
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
                    update(user.id)
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
            update(user.id)
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
                    url = await bot.get_proxy_url(user)
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
            url = await bot.get_proxy_url(user)
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
        update(m_id)
        update(u_id)
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
                update(user.id)
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
        colour = await bot.get_colour(user)
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(mimics) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(as_str(react)))
                await asyncio.sleep(0.2)


class MimicSend(Command):
    name = ["RPSend", "PluralSend"]
    description = "Sends a message using a webhook mimic, to the target channel."
    usage = "<0:mimic> <1:channel> <2:string>"
    no_parse = True
    rate_limit = 0.5

    async def __call__(self, bot, channel, message, user, perm, argv, args, **void):
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        mimics = set_dict(mimicdb, user.id, {})
        prefix = args.pop(0)
        c_id = verify_id(args.pop(0))
        channel = await bot.fetch_channel(c_id)
        guild = channel.guild
        msg = argv.split(None, 2)[-1]
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
        if not admin and ("fun" not in enabled or perm < 1):
            raise PermissionError("Not permitted to send into target channel.")
        if m:
            msg = escape_roles(msg)
            if msg.startswith("/tts "):
                msg = msg[5:]
                tts = True
            else:
                tts = False
            if guild and "logM" in bot.data and guild.id in bot.data.logM:
                c_id = bot.data.logM[guild.id]
                try:
                    c = await self.bot.fetch_channel(c_id)
                except (EOFError, discord.NotFound):
                    bot.data.logM.pop(guild.id)
                    return
                emb = bot.as_embed(message, link=True)
                emb.colour = discord.Colour(0x00FF00)
                action = f"**Mimic invoked in** {channel_mention(channel.id)}:\n"
                emb.description = lim_str(action + emb.description, 2048)
                emb.timestamp = message.created_at
                self.bot.send_embeds(c, emb)
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
            if perm < 1:
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
                with bot.ExceptionSender(message.channel, Exception, reference=message):
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
                        guild = message.guild
                        if guild and "logM" in bot.data and guild.id in bot.data.logM:
                            c_id = bot.data.logM[guild.id]
                            try:
                                c = await self.bot.fetch_channel(c_id)
                            except (EOFError, discord.NotFound):
                                bot.data.logM.pop(guild.id)
                                return
                            emb = await self.bot.as_embed(message, link=True)
                            emb.colour = discord.Colour(0x00FF00)
                            action = f"**Mimic invoked in** {channel_mention(channel.id)}:\n"
                            emb.description = lim_str(action + emb.description, 2048)
                            emb.timestamp = message.created_at
                            self.bot.send_embeds(c, emb)
                        for k in sending:
                            mimic = self.data[k.m_id]
                            await self.updateMimic(mimic, guild=message.guild)
                            name = mimic.name
                            url = mimic.url
                            msg = escape_roles(k.msg)
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
                mimic.url = await bot.get_proxy_url(user)
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
                async with delay(120):
                    # Garbage collector for unassigned mimics
                    i = 1
                    for m_id in tuple(self.data):
                        if type(m_id) is str:
                            mimic = self.data[m_id]
                            try:
                                if mimic.u_id not in self.data or mimic.id not in self.data[mimic.u_id][mimic.prefix]:
                                    self.data.pop(m_id)
                            except:
                                self.data.pop(m_id)
                        if not i % 8191:
                            await asyncio.sleep(0.45)
                        i += 1


class _8Ball(ImagePool, Command):
    description = "Pulls a random image from cdn.nekos.life/8ball, and embeds it."
    database = "8ball"
    name = ["🎱"]

    def __call__(self, channel, flags, **void):
        e_id = choice(
            "Absolutely",
            "Ask_Again",
            "Go_For_It",
            "It_is_OK",
            "It_will_pass",
            "Maybe",
            "No",
            "No_doubt",
            "Not_Now",
            "Very_Likely",
            "Wait_For_It",
            "Yes",
            "Youre_hot",
            "cannot_tell_now",
            "count_on_it",
        )
        url = f"https://cdn.nekos.life/8ball/{e_id}.png"
        if "v" in flags:
            return escape_roles(url)
        self.bot.send_as_embeds(channel, image=url)


class Cat(ImagePool, Command):
    description = "Pulls a random image from thecatapi.com, api.alexflipnote.dev/cats, or cdn.nekos.life/meow, and embeds it. Be sure to check out ⟨WEBSERVER⟩/cats!"
    database = "cats"
    name = ["🐱", "Meow", "Kitty", "Kitten"]
    slash = True

    async def fetch_one(self):
        if random.random() > 2 / 3:
            if random.random() > 2 / 3:
                x = 0
                url = await create_future(nekos.cat, timeout=8)
            else:
                x = 1
        else:
            x = 2
        if x:
            if x == 1 and alexflipnote_key:
                d = await Request("https://api.alexflipnote.dev/cats", headers={"Authorization": alexflipnote_key}, json=True, aio=True)
            else:
                d = await Request("https://api.thecatapi.com/v1/images/search", json=True, aio=True)
            if type(d) is list:
                d = choice(d)
            url = d["file" if x == 1 and alexflipnote_key else "url"]
        return url


class Dog(ImagePool, Command):
    description = "Pulls a random image from images.dog.ceo, api.alexflipnote.dev/dogs, or cdn.nekos.life/woof, and embeds it. Be sure to check out ⟨WEBSERVER⟩/dogs!"
    database = "dogs"
    name = ["🐶", "Woof", "Doggy", "Doggo"]
    slash = True

    async def fetch_one(self):
        if random.random() > 2 / 3:
            if random.random() > 2 / 3:
                x = 0
                url = await create_future(nekos.img, "woof", timeout=8)
            else:
                x = 1
        else:
            x = 2
        if x:
            if x == 1 and alexflipnote_key:
                d = await Request("https://api.alexflipnote.dev/dogs", headers={"Authorization": alexflipnote_key}, json=True, aio=True)
            else:
                d = await Request("https://dog.ceo/api/breeds/image/random", json=True, aio=True)
            if type(d) is list:
                d = choice(d)
            url = d["file" if x == 1 and alexflipnote_key else "message"]
            if "\\" in url:
                url = url.replace("\\/", "/").replace("\\", "/")
            while "///" in url:
                url = url.replace("///", "//")
        return url


class Muffin(ImagePool, Command):
    name = ["🧁", "Muffins"]
    description = "Muffin time! What more is there to say? :D"
    database = "muffins"

    async def fetch_one(self):
        if xrand(3):
            s = await Request(f"https://www.gettyimages.co.uk/photos/muffin?page={random.randint(1, 100)}", decode=True, aio=True)
            url = "https://media.gettyimages.com/photos/"
            spl = s.split(url)[1:]
            imageset = {url + i.split('"', 1)[0].split("?", 1)[0] for i in spl}
        else:
            d = await Request(f"https://unsplash.com/napi/search/photos?query=muffin&per_page=20&page={random.randint(1, 19)}", json=True, aio=True)
            imageset = {result["urls"]["raw"] for result in d["results"]}
        return imageset


class XKCD(ImagePool, Command):
    description = "Pulls a random image from xkcd.com and embeds it."
    database = "xkcd"

    async def fetch_one(self):
        s = await Request("https://c.xkcd.com/random/comic", decode=True, aio=True)
        search = "Image URL (for hotlinking/embedding): "
        s = s[s.index(search) + len(search):]
        url = s[:s.index("<")].strip()
        return url


class Inspiro(ImagePool, Command):
    name = ["InspiroBot"]
    description = "Pulls a random image from inspirobot.me and embeds it."
    database = "inspirobot"

    def fetch_one(self):
        return Request("https://inspirobot.me/api?generate=true", decode=True, aio=True)


class Giphy(ImagePool, Command):
    name = ["GIFSearch"]
    description = "Pulls a random image from a search on giphy.com using tags."
    threshold = 4
    sem = Semaphore(5, 256, rate_limit=1)

    def img(self, tag=None, search_tag=None):
        file = f"giphy~{tag}"

        async def fetch(tag, search_tag):
            resp = await Request(f"https://api.giphy.com/v1/gifs/search?offset=0&type=gifs&sort=&explore=true&api_key={giphy_key}&q={search_tag}", aio=True, json=True)
            images = {entry["images"]["source"]["url"].split("?", 1)[0] for entry in resp["data"]}
            return images

        async def fetchall(tag, search_tag):
            await asyncio.sleep(1)
            images = set()
            for i in range(1, 100):
                async with self.sem:
                    resp = await Request(f"https://api.giphy.com/v1/gifs/search?offset={i * 25}&type=gifs&sort=&explore=true&api_key={giphy_key}&q={search_tag}", aio=True, json=True)
                data = resp["data"]
                if not data:
                    break
                for entry in data:
                    url = entry["images"]["source"]["url"].split("?", 1)[0]
                    images.add(url)
                if len(data) < 25:
                    break
            data = set_dict(self.bot.data.imagepools, file, alist())
            for url in images:
                if url not in data:
                    data.add(url)
                    self.bot.data.imagepools.update(file)
            return images

        if file not in self.bot.data.imagepools.finished:
            create_task(fetchall(tag, search_tag))
        return self.bot.data.imagepools.get(file, fetch, self.threshold, args=(tag, search_tag))
    
    async def __call__(self, bot, channel, flags, args, **void):
        if not args:
            raise ArgumentError("Input string is empty.")
        args2 = ["".join(c for c in full_prune(w) if c.isalnum()) for w in args]
        tag = "%20".join(sorted(args2))
        search_tag = "%20".join(args2)
        url = await self.img(tag, search_tag)
        if "v" in flags:
            return escape_roles(url)
        self.bot.send_as_embeds(channel, image=url)