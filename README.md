# Miza
Welcome to Miza, a multipurpose Discord bot created by [Thomas Xin](https://github.com/thomas-xin). Writing this README is [smudgedpasta](https://github.com/smudgedpasta), Miza's secondary bot owner! :3 ~~I can't speak for the code as much as I can help explain the functionality, so I'll keep things worded simply for everyone to understand...~~

## Table of Contents

Introduction | Command Usage
------------ | -------------
Where can I find what? | (Not written yet)
This explains the basis of where everything is and why for if you ever want to use the code yourself. | (Not written yet)
How do I use the code? | (Not written yet)
This explains the key things to remember when hosting Miza, judging by my own experience. | (Not written yet)

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

Ah yes, the endless spam a few minutes after start-up...

## WIP, I still need to write the rest of the funcionality as well as examples of the commands themselves, but for now the Smudge must go sleepies... zzz