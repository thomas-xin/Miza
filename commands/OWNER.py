import discord
try:
    from smath import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from smath import *


class Execute:
    is_command = True

    def __init__(self):
        self.name = ["Exec", "Eval"]
        self.min_level = nan
        self.description = (
            "Causes all messages in the current channel to be executed as python code on the bot."
            + " WARNING: DO NOT ALLOW UNTRUSTED USERS TO POST IN CHANNEL."
        )
        self.usage = "<enable(?e)> <disable(?d)>"
        self.flags = "ed"

    async def __call__(self, _vars, flags, channel, **void):
        if "e" in flags:
            _vars.updaters["exec"].channel = channel
            return (
                "```css\nSuccessfully changed code channel to "
                + uniStr(channel.id) + ".```"
            )
        elif "d" in flags:
            _vars.updaters["exec"].channel = freeClass(id=None)
            return (
                "```css\nSuccessfully removed code channel.```"
            )
        return (
            "```css\ncode channel is currently set to "
            + uniStr(_vars.updaters["exec"].channel.id) + ".```"
        )
        

class updateExec:
    is_database = True
    name = "exec"
    no_file = True

    def __init__(self):
        self.channel = freeClass(id=None)

    async def __call__(self):
        pass

    async def _nocommand_(self, message, **void):
        _vars = self._vars
        if message.author.id == _vars.client.user.id:
            return
        if message.channel.id == self.channel.id:
            proc = message.content
            while proc[0] == " ":
                proc = proc[1:]
            if proc.startswith("//") or proc.startswith("||") or proc.startswith("\\\\") or proc.startswith("#"):
                return
            if proc.startswith("`") and proc.endswith("`"):
                proc = proc.strip("`")
            if not proc:
                return
            output = None
            try:
                print(proc)
                try:
                    output = eval(proc, _vars._globals)
                except:
                    try:
                        exec(proc, _vars._globals)
                        output = str(proc) + " Successfully executed!"
                    except:
                        output = traceback.format_exc()
                try:
                    if type(output) in (str, bytes):
                        raise TypeError
                    output = tuple(output)
                except TypeError:
                    pass
                if type(output) is tuple:
                    output = await _vars.recursiveCoro(output)
                elif asyncio.iscoroutine(output):
                    output = await output
                await self.channel.send(limStr("```py\n" + str(output) + "```", 2000))
            except:
                await self.channel.send(limStr(
                    "```py\n" + traceback.format_exc().replace("```", "") + "```",
                    2000,
                ))
            if output is not None:
                _vars._globals["output"] = output
                _vars._globals["_"] = output
        elif message.guild is None:
            emb = discord.Embed()
            emb.add_field(
                name=str(message.author) + " (" + str(message.author.id) + ")",
                value=_vars.strMessage(message),
            )
            await self.channel.send(embed=emb)