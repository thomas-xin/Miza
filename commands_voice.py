from smath import *


class join:
    is_command = True

    def __init__(self):
        self.name = ["summon"]
        self.min_level = 1
        self.description = "Summons the bot into a voice channel."
        self.usage = ""

    async def __call__(self, user, _vars, guild, **void):
        voice = user.voice
        vc = voice.channel
        await vc.connect(timeout=_vars.timeout, reconnect=True)
        if vc.id not in _vars.queue:
            _vars.queue[vc.id] = []
        return (
            "Successfully connected to **" + vc.name
            + "** in **" + guild.name + "**."
            )


class leave:
    is_command = True

    def __init__(self):
        self.name = ["quit","dc","disconnect"]
        self.min_level = 1
        self.description = "Leaves a voice channel."
        self.usage = ""

    async def __call__(self, user, client, _vars, **void):
        voice = user.voice
        vchannel = voice.channel
        found = False
        for vclient in client.voice_clients:
            v_id = vclient.channel.id
            if vchannel.id == v_id:
                _vars.queue[v_id] = []
                await vclient.disconnect(force=False)
                return "Successfully disconnected from **" + vchannel.name + "**."
        raise EOFError("Unable to find connected channel.")


class queue:
    is_command = True

    def __init__(self):
        self.name = ["q", "qlist", "play"]
        self.min_level = 0
        self.description = "Shows the queue, or plays a song in voice."
        self.usage = "<link:[]> <verbose:(?v)>"

    async def __call__(self, user, _vars, args, guild, **void):
        voice = user.voice
