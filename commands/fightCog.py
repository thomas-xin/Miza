from cogs.imports import *


class Player:
    def __init__(self, member):
        self.member = member
        self.hp = 100
        self.defense = 0


class Battle(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.occupied = []

    async def attack(self, player):
        damage = int((math.pow(random.randrange(30, 95), 1.35) / 10)
                     * 1 - player.defense / 100)
        player.hp -= damage
        return damage

    async def defend(self, player):
        player.defense += 5
        heal = random.randrange(20, 38)
        player.hp += heal
        if player.hp > 115:
            player.hp = 115
            heal = 0
        if player.defense > 45:
            player.defense = 45
            return heal, True
        return heal, False

    async def turn(self, ctx, p1, p2):
        await ctx.send(f"{p1.member.mention} **choose a move**:  `attack`, `defend`, `escape`")
        try:
            choice = await self.bot.wait_for('message',
                                             check=lambda m: m.channel == ctx.channel and m.author == p1.member and
                                             (
                                                 m.content == "attack" or m.content == "defend" or m.content == "escape"),
                                             timeout=30)
            if choice.content.lower() == "defend":
                healAmount, defenseMaxed = await self.defend(p1)
                if defenseMaxed:
                    await ctx.send(f"You healed for `{healAmount}`, but your defense is maxed out")
                else:
                    await ctx.send(f"You healed for `{healAmount}`, and your defense rose by `5`")
            elif choice.content.lower() == "attack":
                damage = await self.attack(p2)
                await ctx.send(f"You attacked dealing **{damage}** damage")
            elif choice.content.lower() == "escape":
                await ctx.send(f"{p1.member.name} tried escaping. **tried**")
                await ctx.send(embed=discord.Embed(title="CRITICAL HIT", description="9999 Damage!",
                                                   colour=discord.Color.red()))
                p1.hp = -9999

        except asyncio.TimeoutError:
            await ctx.send(f"`{p2.member.name}` got tired of waiting and bonked `{p1.member.name}` on the head.")
            await ctx.send(embed=discord.Embed(title="CRITICAL HIT", description="9999 Damage!",
                                               colour=discord.Color.red()))
            p1.hp = -9999
        await ctx.send(
            f" \n {p1.member.mention} STATS:  **HP:** `{p1.hp}` |  **Defense**: `{p1.defense}`\n \n {p2.member.mention} STATS: **HP**: `{p2.hp}` |  **Defense**: `{p2.defense}` \n")

    @commands.command(aliases=["battle"])
    async def fight(self, ctx, opponent: discord.Member):
        if ctx.channel.id in self.occupied:
            await ctx.send("This battlefield is occupied")
            return
        else:
            self.occupied.append(ctx.channel.id)
        if opponent == ctx.message.author:
            await ctx.send(f"{ctx.author.mention} hurt itself in its confusion.")
            self.occupied.remove(ctx.channel.id)
            return
        if opponent.bot:
            await ctx.send(f"You try fighting the robot.\n\n*pieces of you can be found cut up on the battlefield*")
            self.occupied.remove(ctx.channel.id)
            return
        if (random.randrange(0, 2)) == 0:
            p1 = Player(ctx.message.author)
            p2 = Player(opponent)
        else:
            p1 = Player(opponent)
            p2 = Player(ctx.message.author)
        await ctx.send(embed=discord.Embed(title="Battle",
                                           description=f"""{ctx.author.mention} is challenging {opponent.mention}!
        let the games begin."""))
        await ctx.send(f"{p1.member.mention} got the jump on {p2.member.mention}!")
        toggle = True
        while p1.hp >= 0 and p2.hp >= 0:
            if toggle:
                await self.turn(ctx, p1, p2)
                toggle = False
            else:
                await self.turn(ctx, p2, p1)
                toggle = True

        self.occupied.remove(ctx.channel.id)
        if p1.hp > 0:
            winner = p1
            loser = p2
        else:
            winner = p2
            loser = p1
        case = random.randrange(0, 4)
        if case == 0:
            await ctx.send(f"{winner.member.mention} is having human meat for dinner tonight.")
        if case == 1:
            await ctx.send(f"{winner.member.mention} is dancing on `{loser.member.name}`'s corpse.")
        if case == 2:
            await ctx.send(f"{winner.member.mention} did some good stabbing.")
        if case == 3:
            await ctx.send(f"{winner.member.mention} Is victorious!")


def setup(client):
    client.add_cog(Battle(client))
