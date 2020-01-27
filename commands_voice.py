class join:
    is_command = True

    def __init__(self):
        self.name = ["summon"]
        self.minm = 1
        self.desc = "Summons the bot into a voice channel."
        self.usag = ""

    async def __call__(self, user, _vars, guild, **void):
        voice = user.voice
        vc = voice.channel
        await vc.connect(timeout=_vars.timeout, reconnect=True)
        if vc.id not in _vars.queue:
            _vars.queue[vc.id] = []
        return (
            "Successfully connected to **" + vc.name
            + "** in **" + guild + "**."
            )


class queue:
    is_command = True

    def __init__(self):
        self.name = ["q", "qlist", "play"]
        self.minm = 0
        self.desc = "Shows the queue, or plays a song in voice."
        self.usag = "<link:[]> <verbose:(?v)>"

    async def __call__(self, user, _vars, args, guild, **void):
        voice = user.voice
