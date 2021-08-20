# [Miza](http://mizabot.xyz)
Welcome to Miza, a multipurpose Discord bot created by [Thomas Xin](https://github.com/thomas-xin). Writing this README is [smudgedpasta](https://github.com/smudgedpasta), Miza's secondary bot owner! :3

![Miza](https://github.com/thomas-xin/Miza/blob/e62dfccef0cce3b0fc3b8a09fb3ca3edfedd8ab0/misc/title-rainbow.gif?raw=true)

## Table of Contents

Headings | Explanations
------------ | -------------
[Where can I find what?](#p1) | Will talk you through where everything within the code files can be found!
[How do I use the code?](#p2) | Will talk you through the basics of how to host the code, covering potential error-prone areas!
[Support!](#p3) | Links to where you can find Miza and get support!

<a id="p1"></a>
## Where can I find what?

First and foremost, the front folder here contains all your generic license, requirements, etc... (Though requirements is necessary for the *install_update* funtionality which, automatically checks for and installs any missing modules.) But most significantly, the main files responsible for running the bots code. Throughout the code, you will frequently see `from common import *`, which is because *common.py* contains all the main necessary functions and imports to be used throughout. *main.py* is the main process, while everything else runs as a subprocess, so if you make any changes to main.py, it'll require a manual restart. Most of the bots optimization and data collection funtionaility can be found in these files, (such as running the bot of course, starting the heartbeat.tmp and other log related code, caching, assigning variables of the Github Directory link, default bot prefix, etc...) As for where things are located...

- **commands**

You may think its unnecessary to explain all this, but before I learnt my way around, I got so lost in all of Miza's files, so hopefully this is helpful. The commands folder speaks for itself, all of the bots command categories can be found in here, and they are located in the same way they are categorized under ~help.

- **misc**

Misc contains all the different files that the bot needs to pull from, such as the avatar (which gets automatically uploaded to the Discord Developers Portal the first time the code is run), the rainbow bar emojis (which get automatically uploaded to a server Miza is in if it cannot find the emojis already), the code necessary for converting org files, computing math equations, finding timezones and etc. You can change the bots avatar and emojis if you want to; *but if you want the code to use them the same way, the filename must be kept the same.*


<a id="p2"></a>
## How do I use the code?

I'm just going to comment on what I personally found to be the most important things to know when hosting Miza. First of all, download this heccin chonka of a directory. How Miza is ran currently is through an *auth.json*, which automatically gets created if Miza cannot locate the file. Alternatively, as of 11/14/2020, the general layout can be found at the top of *main.py* if you wish to copy it exactly. **This file is necessary, as the bot cannot run without the token, and some features may not work correctly without their API Key (obviosuly).** If you've successfully run the bot, you'll see some new folders in your front folder here. The most important to acknowledge are *saves* and *backup*. The saves folder is the entire databse, ~~enter with caution because wow if my file explorer doesn't hate loading this...~~ The backup folder automatically saves the current database to a zip file, going by date. If you want to export the database somewhere, the quickest way to do so is to just get rid of the day's backup zip, Miza will make a new one within a couple of minutes. Its what Thomas and I do. 🙃 Now to address some issues I've personally had hosting Miza, and solutions for if anybody experiences the same...

- **Dependencies...**

Miza should automatically install all the dependencies necessary the first time you run her, so don't you worry about chasing after everything in *requirements.txt*. Miza will take whatever is in that file and download it all for you. As of 08/12/2021, FFmpeg will also be installed when you first run Miza. During this first run, the program is also going to look through the servers it is in to seek out open candidates for placing some emojis that Miza will also require for certain UI features, so make sure you have a good space set up too. Finally, all API Keys that Miza will require as located in *auth.json* you will have to obtain yourself.

- **Voice commands not working?**

Make sure you have [*FFmpeg*](https://www.ffmpeg.org/) and [*Python*](https://www.python.org/downloads/) installed onto your computer and in your PATH (it doesn't need to be in the same directory as Miza). I uh... Actually have my ffmpeg pathed by pathing to the misc folder found in [Miza Player](https://github.com/thomas-xin/Miza-Player) (an awesome program you should definitely try out!) 🙃

![ffmpeg](https://cdn.discordapp.com/attachments/688253918890688521/777473182294474753/image0.png)

**Note that the voice commands run in a subprocess concurrently to the main program. If you are still facing issues, this may be the cause, and I suggest asking for support from Thomas Xin.**

- **MemoryError()**

This used to be an issue caused by Miza trying to cache too much at one time. This caching system has been watered down significantly since the time this README was made, so this wont be as likely of an issue anymore. But in case it is still a problem, here's how to fix it. Head on over to the top of *bot.py*, and look for the following function:

```py
def __init__(self, cache_size=4194304, timeout=24):
        self.start_time = utc()
        super().__init__(max_messages=256, heartbeat_timeout=60, guild_ready_timeout=5, intents=self.intents)
        self.cache_size = cache_size
        self.cache = fcdict({c: {} for c in self.caches})

        # Code continues...
```

Just reduce the number in `cache_size=4194304` and you should be good to go.

**Note that this memory cache gets cleared upon reset, and Miza has a seperate disk cache that doesn't upon reset, but all files in the caches get automtically cleared after 2 weeks.**

- **IP Address exposure**

Miza used to host a few Minecraft Servers which is why this feature used to be a doxx moment for me. That is no longer a risk, but Miza will still obtain your IP Address to store it internally for features such as the webserver. If you don't want this, go back to *bot.py* and look for the `get_ip()` function at around line 1475. It should look like this:

```py
async def get_ip(self):
    resp = await Request("https://api.ipify.org", decode=True, aio=True)
    self.update_ip(resp)
```

Change `resp = await Request("https://api.ipify.org", decode=True, aio=True)` to `resp = "\u200b"` and it'll take your IP as `None`.

- **Where does Miza log?**

Miza logs up to three places: A *log.txt* (which the file gets refreshed upon restart), the console where you're running the code (we just use a *main.bat* file to run Miza through the Command Prompt on Windows usually, as none of Miza's subprocesses can run on Linux) and a log within Discord itself (which isn't hardcoded, you can enable it as displayed below.)

![Screenshot2](https://cdn.discordapp.com/attachments/727087981285998593/777554361769000960/Capture10.PNG)

![Screenshot3](https://cdn.discordapp.com/attachments/727087981285998593/777554360859099146/Capture9.PNG)

![Screenshot1](https://cdn.discordapp.com/attachments/688253918890688521/804652403445727272/unknown.png)

![Screenshot4](https://cdn.discordapp.com/attachments/727087981285998593/777554358095183893/Capture8.PNG)

- **Why wont Miza work on Linux?**

Linux doesn't like her subprocesses, this has been an issue since the beginning. If you wish to host Miza, it is best you do so from a Windows OS, as Miza's infrastructure is designed on and for Windows.

<a id="p3"></a>
## Support!

With that concludes the basic introduction of hosting Miza. The code is commented and explaining where everything is, so feel free to explore further to see what you can change, and if you have any questions, [Thomas Xin](https://github.com/thomas-xin) is your guy to ask!

Miza has so much functionality, with so much to explore, with all the command categories being: **voice**, **fun**, **admin**, **image**, **string** and **main**. Miza's dedicated creator loves to find ways to make Miza bigger and better all the time, with lots of code optimization and fun to be had!

[Read our Wiki!](https://github.com/thomas-xin/Miza/wiki)

[Check out our Website!](http://mizabot.xyz)

*The the domain redirects to a webserver which includes: a command tester within the comforts of your browser, documentation on the API, a free file host, and more!*

[Join our Discord Support Server!](https://discord.gg/cbKQKAr)
