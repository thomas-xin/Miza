# [Miza](http://27.33.133.250:9801)
Welcome to Miza, a multipurpose Discord bot created by [Thomas Xin](https://github.com/thomas-xin). Writing this README is [smudgedpasta](https://github.com/smudgedpasta), Miza's secondary bot owner! :3 ~~I can't speak for the code as much as I can help explain the functionality, so I'll keep things worded simply for everyone to understand...~~

![Miza](https://cdn.discordapp.com/attachments/688253918890688521/777456623555051521/image0.png)

## Table of Contents

Introduction | Discord Usage
------------ | -------------
Where can I find what? | Developer tools within Discord
This explains the basis of where everything is and why for if you ever want to use the code yourself. | This explains what the bot owner(s) can do with Miza.
How do I use the code? | Command syntax and flags
This explains the key things to remember when hosting Miza, judging by my own experience. | This explains the structure of Miza's commands and different ways to use them properly.

## Introduction

```
Where can I find what?
```

First and foremost, the front folder here contains all your generic license, requirements, etc... (Though requirements is necessary for the *install_update* funtionality which, automatically checks for and installs any missing modules.) But most significantly, the main files responsible for running the bots code. Throughout the code, you will frequently see `from common import *`, which is because *common.py* contains all the main necessary functions and imports to be used throughout. *main.py* is the main process, while everything else runs as a subprocess, so if you make any changes to main.py, it'll require a manual restart. Most of the bots optimization and data collection funtionaility can be found in these files, (such as running the bot of course, starting the heartbeat.tmp and other log related code, message/attachement caching, assigning variables of the Github Directory link, default bot prefix, etc...) As for where things are located...

- **commands**

You may think its unnecessary to explain all this, but before I learnt my way around, I got so lost in all of Miza's files, so hopefully this is helpful. The commands folder speaks for itself, all of the bots command categories can be found in here, and they are located in the same way they are categorized under ~help.

- **misc**

Misc contains all the different files that the bot needs to pull from, such as the avatar (which gets automatically uploaded to the Discord Developers Portal the first time the code is ran), the rainbow bar emojis (which get automatically uploaded to a server Miza is in if it cannot find the emojis already, the code necessary for converting org files, computing math equations, finding timezones and etc. You can change the bots avatar and emojis if you want to; *but if you want the code to use them the same way, the filename must be kept the same.*

```
How do I use the code?
```

I'm just going to comment on what I personally found to be the most important things to know when hosting Miza. First of all, download this heccin chonka of a directory. How Miza is ran currently is through an *auth.json*, which as of 14/11/2020 (UTC), the general layout can be found at the top of *main.py* if you wish to host a Miza of your own and copy it exactly. **This file is necessary, as the bot cannot run without its token (obviosuly).** If you've successfully run the bot, you'll see some new folders in your front folder here. The most important to acknowledge are *saves* and *backup*. The saves folder is the entire databse, ~~enter with caution because wow if my file explorer doesn't hate loading this...~~ The abckup folder automatically saves the current database to a zip file, going by date. If you want to export the database somewhere, the quickest way to do so is to just get rid of the day's backup zip, Miza will make a new one within a couple of minutes. Its what Thomas and I do. 🙃 Now to address some issues I've personally had hosting Miza, and solutions for if anybody experiences the same...

- **MemoryError()**

Ah yes, the endless spam a few minutes after start-up... This is an issue I ran into initially, caused by Miza trying to cache too much at one time. Usually Miza only caches things in chunks if its necessary (like someone running a command on an attachement sent years ago for example). You can counter this issue by reducing how much content Miza caches, which is found right at the top of *bot.py*, in a function that looks like this:

```py
def __init__(self, cache_size=4194304, timeout=24):
        # Initializes client (first in __mro__ of class inheritance)
        self.start_time = utc()
        super().__init__(max_messages=256, heartbeat_timeout=60, guild_ready_timeout=5, intents=self.intents)
        self.cache_size = cache_size
        # Base cache: contains all other caches
        self.cache = fcdict({c: {} for c in self.caches})

        # Code continues...
```

Just reduce the number in `cache_size=4194304` and you should be good to go.

- **IP Address exposure**

So, the main Miza bot hosts a few Minecraft Servers, and in order to keep people up-to-date with the IP whenever there's a change, the ~status will show your IP Address. If you don't want your IP Address exposed publicly, you can change this in the same *bot.py* file as before, down in the `get_ip()` function at around line 1475. It should look like this:

```py
async def get_ip(self):
    resp = await Request("https://api.ipify.org", decode=True, aio=True)
    self.update_ip(resp)
```

Change `resp = await Request("https://api.ipify.org", decode=True, aio=True)` to `resp = "\u200b"` and it'll always appear as `None`.

- **OSError()**

Alright, to quote this issue from when Thomas explained it to me...
> Invalid argument as a windows error (which is why it's OS error) means that the process being selected is invalid, which in this case, is caused by miza trying to send data to another process running on the computer that was closed or otherwise not open. The image and math commands (and in the latest version of miza, the webserver) run in separate processes entirely, in order to share CPU more fairly and not clog up the main bot when being used for time consuming operations. Because of the matplotlib compatibility issue with python 3.9, I had to effectively make miza run two different python versions, 3.9.0 and 3.8.5, because I'd already updated a lot to 3.9. So... in order to make that possible, I added a "python path" variable to my auth.json, which only worked for Miza. The latest version of miza should run perfectly fine now with python_path set to ""

So in a nutshell, make sure you have `"python_path":"",` in your auth.json, or else you wont be able to use any voice commands, image commands, or etc.

- **Voice commands still not working?**

Make sure you have *ffmpeg* installed onto your computer and in your PATH (it doesn't need to be in the same directory as Miza). I uh... Actually have my ffmpeg pathed by pathing to the misc folder found in [Miza Player](https://github.com/thomas-xin/Miza-Player). 🙃

![ffmpeg](https://cdn.discordapp.com/attachments/688253918890688521/777473182294474753/image0.png)

With that concludes the basic introduction of hosting Miza. The code is commented and explaining where everything is, so feel free to explore further to see what you can change, and if you have any questions, [Thomas Xin](https://github.com/thomas-xin) is your guy to ask!

## Discord Usage

```
Developer tools within Discord
```

So Miza isn't just your standard Discord bot. It can go as far as opening a Python terminal within Discord itself, allowing you a discord.py console of Discord API in the comforts of your own server, with a custom cache that Thomas created. For a better reference of discord.py, visit the [Discord.py official documentation!](https://discordpy.readthedocs.io/en/latest/) (Note that Miza will always be able to pick up on edited messages as well, in both the terminal and with any commands.)

#### ⚠ **IMPORTANT NOTE** ⚠
**The in-Discord terminal is *very* powerful, allowing someone to have more power and risk of damaging your servers than someone getting their hands on the actual bot token itself. This being said, be very careful with who you intrust bot ownership too. Below are a but a few of the most basic examples of what the terminal can do.**

![Screenshot](https://cdn.discordapp.com/attachments/727087981285998593/777536790574923786/unknown.png)

![Screenshot2](https://cdn.discordapp.com/attachments/727087981285998593/777539347884933150/Capture3.PNG)

![Screenshot3](https://cdn.discordapp.com/attachments/727087981285998593/777539328062259200/Capture2.PNG)

![Screenshot4](https://cdn.discordapp.com/attachments/727087981285998593/777542932139081738/Capture6.PNG)

![Screenshot5](https://cdn.discordapp.com/attachments/727087981285998593/777544002957738054/unknown.png)

Miza also logs up to three places: A *log.txt* (which the file gets refreshed upon restart), the console where you're running the code (we just use a *main.bat* file to run Miza through the Command Prompt on Windows usually, as none of Miza's subprocesses can run on Linux) and a log within Discord itself (which isn't hardcoded, you can enable it similarly to how I enabled the Python terminal above.)

![Screenshot6](https://cdn.discordapp.com/attachments/727087981285998593/777554361769000960/Capture10.PNG)

![Screenshot7](https://cdn.discordapp.com/attachments/727087981285998593/777554360859099146/Capture9.PNG)

![Screenshot8](https://cdn.discordapp.com/attachments/727087981285998593/777554358095183893/Capture8.PNG)

Now onto more command-based things...

- **What are trusted servers?**

Certain commands can be quite CPU consuming, especially used in multiple places. Because of this, "trusted servers" exist, which is essentially a server that has all access to commands. When Miza first joins a server, it wont by default (unless the server has been already added to the trusted list), and it is down to the bot owner to assign these.

#### **Note: This feature is removed from the current build of Miza, however, anyone with inf/nan permissions will have to enable command categories per channel to still have access to all commands if desired.**

![Screenshot9](https://cdn.discordapp.com/attachments/727087981285998593/777882312724709386/unknown.png)

- **What is the blacklist for?**

The blacklist is used to prevent users from being able to use the bot's commands, and any DM's they send, commands they try to run, etc, will not be logged. When a user attempts to DM Miza, it will reply in the way Discord would usually respond if you tried to DM someone who has blocked you.

![Screenshot10](https://cdn.discordapp.com/attachments/727087981285998593/777883584484605972/Capture1.PNG)

![Screenshot11](https://cdn.discordapp.com/attachments/727087981285998593/777883566989901834/Capture.PNG)

```
Command syntax and flags
```

By default, Miza's prefix is a tilde character, not be confused with a dash character. It looks like this: `~` But you can also assign a new prefix (on a per-server basis) with the ~prefix command. Alternatively, you can @ Miza or use ~miza/(your prefix)miza which would serve as an alias for all bot prefixes. It does not matter how far away the command is from the prefix!

![Screenshot12](https://cdn.discordapp.com/attachments/687567112952348746/779058870475030579/unknown.png)

To enhance or change the effect of a command, Miza has a flag functionality. Flags can be used with either a dash (`-`), a plus (`+`) or a question mark (`?`), it does not matter which, they do the same thing. Usually, they are the letters a, f, or v, and would follow either directly after the command or after you've given Miza an argument to work with. Use the ~help command if you're confused and would like to see what flag goes with what or where. Here are some examples of how you can use flags!

![Screenshot13](https://cdn.discordapp.com/attachments/687567100767633432/779061024414367784/image0.png)

![Screenshot14](https://cdn.discordapp.com/attachments/687567100767633432/779087453188259891/image0.png)

![Screenshot15](https://cdn.discordapp.com/attachments/687567100767633432/779063716317888582/Capture.PNG)

![Screenshot16](https://cdn.discordapp.com/attachments/687567100767633432/779063717957599312/Capture1.PNG)

It is also worth mentioning that Miza has a math functionaility in not just a math command, and thanks to it, it can parse some complicated inputs, ranging from decimals/floats too... Brayconn. 🙃 (In the below example, he is using the ~loop command, which lets you repeat a command an inputted certain amount of times.)

![Screenshot17](https://cdn.discordapp.com/attachments/687567100767633432/779065528551473182/unknown.png)

On the topic of parsing, Miza can also interpret unciode characters, has a lookup feature which finds the most accurate match (so you do not have to write the user's name exactly and/or @ them) can lookup users from outside of the guild via user ID, understand commands without being case sensitive, and etc!

![Screenshot18](https://cdn.discordapp.com/attachments/687567100767633432/779070391657562162/image0.png)

![Screenshot19](https://cdn.discordapp.com/attachments/687567100767633432/779070391968202792/image1.png)

![Screenshot20](https://cdn.discordapp.com/attachments/687567100767633432/779071275091886081/unknown.png)

Finally, I will address how permissions and levels work. People can use certain commands with Miza based on what perms they have. Bot owners will default to having "nan" perms, nan meaning nothing, which is the strongest levels of perms someone can have, as it bypasses perm restrictions entirely. After that is "inf", which is typically the level an admin or a server owner will have, but someone with nan perms can assign someone to inf (or any level), and server owners can set anyone's level to inf and under. After inf is level 3, then level 2, then level 1, then level 0, level 0 being the lowest level of perms to Miza commands and 3 being the highest under moderator level.

![Screenshot21](https://cdn.discordapp.com/attachments/687567100767633432/779074382063992852/image0.png)

You may notice that sometimes Miza will react a ✨ on your message, too! Miza has a levelling system, ~~don't worry, its not as invasive as certain other bots infamous level-up @, Miza doesn't send a message when you level up at this current time.~~ If you recieve this reaction, then congrats, you've just had a huge boost of exp and other rewards! You can gain rewards through various FUN categories, ranging from an in-Discord 2048 game (which lets you adjust the size of the playing board) to a slots machine.

- **Okay, but how do I make a command?**

Its quite simple! Miza's commands, like almost any Discord bot's that are in multiple files, are written in classes. Below is an example of a fun little test command written by our awesome collaborator, GDB-spur, as an example!

```py
class muffin(Command):
    name=["muffins", "🧁", "muffin"]
    description: "Muffin time! What more is there to say? :D"
    
    def __call__(self, **void):
        return "Muffin time :D 🧁🧁🧁"
```

Miza has so much functionality, with so much to explore, with all the command categories being: **voice**, **fun**, **admin**, **image**, **string** and **main**. Miza's dedicated creator loves to find ways to make Miza bigger and better all the time, with lots of code optimization and fun to be had, with voice commands capable of effects like bassboosting, reverbing and adding a nightcore/resample filter at any intensity you want, (*completely* for free with no premium permmissions needed and 4 different downloaders to keep YouTube links to play or download stable) to image commands allowing you to blend images together at any blend mode you'd like or even make a rainbow gif out of them! This README barely scratches the surface, so again, any questions about Miza's functionality should be asked to Thomas Xin!

[Read our Wiki!](https://github.com/thomas-xin/Miza/wiki)

[Join our Discord Server!](https:/discord.gg//cbKQKAr)
