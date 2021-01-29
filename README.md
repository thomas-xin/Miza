# [Miza](http://mizabot.xyz)
Welcome to Miza, a multipurpose Discord bot created by [Thomas Xin](https://github.com/thomas-xin). Writing this README is [smudgedpasta](https://github.com/smudgedpasta), Miza's secondary bot owner! :3

![Miza](https://cdn.discordapp.com/attachments/688253918890688521/777456623555051521/image0.png)

## Table of Contents

Headings | Explinations
------------ | -------------
Where can I find what? | Will talk you through where everything within the code files can be found!
How do I use the code? | Will talk you through the basics of how to host the code, covering potential error-prone areas!
Support! | Links to where you can find Miza and get support!

## [Where can I find what?](https://github.com/thomas-xin/Miza/tree/master/commands)

First and foremost, the front folder here contains all your generic license, requirements, etc... (Though requirements is necessary for the *install_update* funtionality which, automatically checks for and installs any missing modules.) But most significantly, the main files responsible for running the bots code. Throughout the code, you will frequently see `from common import *`, which is because *common.py* contains all the main necessary functions and imports to be used throughout. *main.py* is the main process, while everything else runs as a subprocess, so if you make any changes to main.py, it'll require a manual restart. Most of the bots optimization and data collection funtionaility can be found in these files, (such as running the bot of course, starting the heartbeat.tmp and other log related code, message/attachement caching, assigning variables of the Github Directory link, default bot prefix, etc...) As for where things are located...

- **commands**

You may think its unnecessary to explain all this, but before I learnt my way around, I got so lost in all of Miza's files, so hopefully this is helpful. The commands folder speaks for itself, all of the bots command categories can be found in here, and they are located in the same way they are categorized under ~help.

- **misc**

Misc contains all the different files that the bot needs to pull from, such as the avatar (which gets automatically uploaded to the Discord Developers Portal the first time the code is ran), the rainbow bar emojis (which get automatically uploaded to a server Miza is in if it cannot find the emojis already), the code necessary for converting org files, computing math equations, finding timezones and etc. You can change the bots avatar and emojis if you want to; *but if you want the code to use them the same way, the filename must be kept the same.*

## How do I use the code?

I'm just going to comment on what I personally found to be the most important things to know when hosting Miza. First of all, download this heccin chonka of a directory. How Miza is ran currently is through an *auth.json*, which as of 14/11/2020 (UTC), the general layout can be found at the top of *main.py* if you wish to host a Miza of your own and copy it exactly. **This file is necessary, as the bot cannot run without its token (obviosuly).** If you've successfully run the bot, you'll see some new folders in your front folder here. The most important to acknowledge are *saves* and *backup*. The saves folder is the entire databse, ~~enter with caution because wow if my file explorer doesn't hate loading this...~~ The backup folder automatically saves the current database to a zip file, going by date. If you want to export the database somewhere, the quickest way to do so is to just get rid of the day's backup zip, Miza will make a new one within a couple of minutes. Its what Thomas and I do. 🙃 Now to address some issues I've personally had hosting Miza, and solutions for if anybody experiences the same...

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

- **Where does Miza log?**

Miza logs up to three places: A *log.txt* (which the file gets refreshed upon restart), the console where you're running the code (we just use a *main.bat* file to run Miza through the Command Prompt on Windows usually, as none of Miza's subprocesses can run on Linux) and a log within Discord itself (which isn't hardcoded, you can enable it as displayed below.)

![Screenshot2](https://cdn.discordapp.com/attachments/727087981285998593/777554361769000960/Capture10.PNG)

![Screenshot3](https://cdn.discordapp.com/attachments/727087981285998593/777554360859099146/Capture9.PNG)

![Screenshot1](https://cdn.discordapp.com/attachments/688253918890688521/804652403445727272/unknown.png)

![Screenshot4](https://cdn.discordapp.com/attachments/727087981285998593/777554358095183893/Capture8.PNG)

## [Support!](http://27.33.133.250:9801/)

With that concludes the basic introduction of hosting Miza. The code is commented and explaining where everything is, so feel free to explore further to see what you can change, and if you have any questions, [Thomas Xin](https://github.com/thomas-xin) is your guy to ask!

Miza has so much functionality, with so much to explore, with all the command categories being: **voice**, **fun**, **admin**, **image**, **string** and **main**. Miza's dedicated creator loves to find ways to make Miza bigger and better all the time, with lots of code optimization and fun to be had!

[Read our Wiki!](https://github.com/thomas-xin/Miza/wiki)

[Check out our Website!](http://mizabot.xyz)

[View our Miza Atlas for a command tester and in-depth command list in the comforts of your browser!](http://mizabot.xyz/mizatlas)

[Join our Discord Support Server!](https:/discord.gg//cbKQKAr)