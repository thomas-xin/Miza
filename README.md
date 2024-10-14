<h1 align="center">
  <img src="https://raw.githubusercontent.com/thomas-xin/Image-Test/master/title-rainbow.webp">
  <br>
  <a href="https://mizabot.xyz">Miza</a>
</h1>


<h4 align="center">A multipurpose Discord bot created by <a href="https://github.com/thomas-xin">Thomas Xin</a>.</h4> <br>

<p align="center">
<a href="https://github.com/thomas-xin/Miza/network/members">
<img alt="GitHub Forks" src="https://img.shields.io/github/forks/thomas-xin/Miza?style=plastic&label=Forks">
</a>

<a href="https://github.com/thomas-xin/Miza/stargazers">
<img alt="GitHub Stars" src="https://img.shields.io/github/stars/thomas-xin/Miza?label=Stars&style=plastic">
</a>

<a href="https://github.com/thomas-xin/Miza/issues">
<img alt="GitHub Issues" src="https://img.shields.io/github/issues/thomas-xin/Miza?style=plastic&label=Issues">
</a>

<a href="https://github.com/thomas-xin/Miza/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/thomas-xin/Miza?style=plastic&label=Contributors">
</a>

<a href="https://github.com/thomas-xin/Miza/blob/master/LICENSE.md">
<img alt="GitHub License" src="https://img.shields.io/github/license/Thomas-Xin/Miza?style=plastic&label=License">
</a>

</p>

<p align="center">
Miza's dedicated creator loves to find ways to make Miza bigger and better all the time, with lots of code, optimization
and fun to be had! with Features Including: <b>Moderation, Music, Fun, Image Manipulation, and more!</b>
</p>

<p align="center">
Please support the development of Miza here if you enjoy her services, or would like access to her premium features!<br>
https://ko-fi.com/mizabot<br>
Miza has been free to use for 3 years now since the start of her development, but is quickly becoming more expensive to maintain. Any donations are greatly appreciated, and 100% will go towards supporting Miza's future!
</p>

<h2 align="center">
Table of Contents
</h2>

Sections     | Explanations
------------ | -------------
[Hosting Miza!](#Hosting-Miza)        | Will talk you through the basics of how to host the code, covering potential error-prone areas!
[AI Features](#AI-Features)           | List of supported AI features, as well as requirements for hosting!
[Folder Structure](#Folder-Structure) | Will talk you through where everything within the code files can be found!
[Support!](#Support)                  | Links to where you can find Miza and get support!
[Credits](#Credits)                   | Credits to all contributors!

<br>

# Notice:
This README is currently out-of-date relative to development of this project, and will not be updated until the current intensive phase of maintenance is complete. Please refer to the support Discord server for any required technical support and questions!

<br>

<a id="Hosting-Miza"></a>
## Hosting Miza

If you would like to host a personal copy of Miza for various reasons, the below information may be helpful to be aware of!

#### Prerequisites:

* [Git](https://git-scm.com/downloads)
* [Python](https://www.python.org/downloads/)
* [Discord Bot Token](https://discord.com/developers/) (_Choose your Desired Bot then go to **Bot** and under Token click **Copy**_)

As of 26 September 2021, Miza now fully works on Windows and Linux. If you're planning to run Miza through a Docker container, there is a working [Dockerfile](https://cdn.discordapp.com/attachments/731709481863479436/891402916265594980/Dockerfile) provided that you may use as a template. Note that if you choose to use this, all other installation steps may be skipped, as said Dockerfile includes all necessary installation preparations.

### Installing Miza

* Launch Git Bash.

* Navigate to your desired directory, by typing this command in Git Bash (or Linux terminal):
```bash
cd [Desired Directory]
Example: cd C:/Users/Miza/Documents/Projects/
```

* If on Linux, the following must be installed (use `yum install` on other distributions of Linux as required):
```bash
sudo apt install git
sudo apt install python3 python3-pip python3-tk
sudo apt install libgmp-dev libmpfr-dev libmpc-dev
```

* Clone this Repository, by typing this command:
```bash
git clone https://github.com/thomas-xin/Miza.git
```

* Run `main.bat` the bot will automatically create an `auth.json` file.

* Fill in `auth.json` file (prefix, discord_token, etc)

* Run `main.bat` again.  _(Miza should now start successfully, and be ready for use. If not read the [next section](#common-issues))_ 

<br>

<a id="Common-Issues"></a>
### Common Issues & FAQ:

<a id="Dependencies"></a>
#### Dependencies:

* Miza should automatically download and install the required dependencies immediately after running her for the first time, notably the Python modules and FFmpeg. If this fails despite multiple attempts, you may manually invoke the installation using `pip install -r requirements.txt` or any variant that works on your device.

> _As of 08/12/2021, FFmpeg will be automatically installed after first run, Miza is also going to look through the servers it is in to seek out open candidates for placing some emojis that Miza will also require for certain UI features, so make sure you have a good space set up too. Finally, all API Keys that Miza will require as located in *auth.json* you will have to obtain yourself._

<a id="Memory-Requirements"></a>
#### Memory requirements:
* The main Discord bot uses around 2GB of CPU RAM, which increases depending on the amount of Discord servers being loaded.
* Most other features utilise minimal amounts of RAM, although subprocesses may temporarily use a few extra GB during heavy loads (such as file conversion)
* VRAM (GPU RAM) requirements vary depending on the ML models invoked. Depending on demand up to 100GB may be utilised at a time.
* For comparison, the official Miza currently runs on 112GB of GDDR6X, 24GB of GDDR6, 64GB of DDR5, and 192GB of NVMe swap.

<a id="Logs"></a>
#### Logs

Miza logs information in up to three places:
* `log.txt` (which the file gets refreshed upon restart)<sup><a href="#logtxt-image">1</a></sup>
* The Console from which Miza is running.<sup><a href="#consolelog-image">2</a></sup>
* Log within Discord itself (which isn't hardcoded, you can enable it as displayed below) <sup><a href="#discordlog-image">3</a></sup>

_<a id="logtxt-image"><sup><a href="#logtxt-image">[1]</a></sup> Image showing an example of `log.txt` file:</a>_ 

![Screenshot1](https://cdn.discordapp.com/attachments/727087981285998593/777554361769000960/Capture10.PNG)

_<a id="consolelog-image"><sup><a href="#consolelog-image">[2]</a></sup> Image showing an example of a Console running Miza:</a>_ 

![Screenshot2](https://cdn.discordapp.com/attachments/727087981285998593/777554360859099146/Capture9.PNG) 

_<a id="discordlog-image"><sup><a href="#discordlog-image">[3]</a></sup> Image showing How to enable log in Discord:</a>_ 

![Screenshot3](https://cdn.discordapp.com/attachments/688253918890688521/804652403445727272/unknown.png)
![Screenshot4](https://cdn.discordapp.com/attachments/727087981285998593/777554358095183893/Capture8.PNG) 

_Note: execute the command `~exec -e log` to enable discord logging on Discord. For debugging purposes, this log may include sensitive information being passed through the bot. Please make sure this is not exposed publicly and/or in production._

<br>

<a id="Folder-Structure"></a>
## Folder Structure

#### Project Directory and cache:

* Has all your generic license, requirements, etc... (Though requirements is necessary for the *install_update* functionality which, automatically checks for and installs any missing modules.) But most significantly, the main files responsible for running the bots code. Throughout the code, you will frequently see `from common import *`, which is because *common.py* contains all the main necessary functions and imports to be used throughout. *main.py* is the main process, while everything else runs as a subprocess, so if you make any changes to main.py, it'll require a manual restart. Most of the bots optimization and data collection functionality can be found in these files, (such as running the bot of course, starting the heartbeat.tmp and other log related code, caching, assigning variables of the GitHub Directory link, default bot prefix, etc...) As for where things are located...
* Unimportant cache data is typically stored in `/cache`. This folder may be deleted in its entirety without problems, although it may cause issues with commands being run in the moment.
* Miza has support for offloading AI model data to different folders, or different drives. To do this, manage the `"cache_path"` key in `auth.json`. This overrides the location models from Huggingface and co. are downloaded to, and may be useful if you would like to preserve space on the C drive. If not specified, Huggingface will typically default to `~/.cache` or `C:/Users/{username}/.cache`.
* The `saves` folder stores saved data from the database. This folder is important for maintaining state, from custom user profiles to linked hosted files.

#### Commands:

* Command categories may be enabled per channel via the `~ec` command, or for the entire bot by removing the `.py` files within the `commands` folder. These modules may be hot swapped through the owner-only `~reload` and `~unload` commands, however please be aware that there may be inconsistencies in behaviour when doing this. Some internal features and databases are also tied to individual commands.

#### Misc:

* Misc has all the different files that the bot needs to pull from, such as the avatar (which gets automatically uploaded to the Discord Developers Portal the first time the code is run), the rainbow bar emojis (which get automatically uploaded to a server Miza is in if it cannot find the emojis already), the code necessary for converting org files, computing math equations, finding time zones and etc. You can change the bots avatar and emojis if you want to; *but if you want the code to use them the same way, the filename must be kept the same.*

<br>

<a id="Support"></a>
## Support!

With that concludes the basic introduction of hosting Miza. The code is commented and explaining where everything is, so feel free to explore further to see what you can change, and if you have any questions, [Thomas Xin](https://github.com/thomas-xin) is your guy to ask! Please be aware that not every platform and use case is actively supported, and with only one main developer there are no guarantees all requests will be fulfilled.

[Read our Wiki!](https://github.com/thomas-xin/Miza/wiki) • [Join our Discord Support Server!](https://discord.gg/cbKQKAr) • [Check out our Website!](http://mizabot.xyz)

*The domain redirects to a webserver which includes: a command tester within the comforts of your browser, a free file host, and more!*
- The website was mostly designed by [Hucario](https://github.com/hucario), who has also been a large help in making this project possible!


<a id="API"></a>
## API

Miza has a public API that has now been moved to [RapidAPI](https://rapidapi.com/thomas-xin/api/miza). Currently not all endpoints and possible features in the API are accessible from there, but it will be updated over time.


<a id="Credits"></a>
## Credits

Thanks to all the [contributers](https://github.com/thomas-xin/Miza/graphs/contributors), for helping Miza grow!

*Helping with this README is [Illouminant](https://github.com/Illouminant), Miza's secondary bot owner!*

<br>

<p align="center">
Please support the development of Miza here if you enjoy her services, or would like access to her premium features!<br>
https://ko-fi.com/mizabot<br>
Miza has been free to use for 3 years now since the start of her development, but is quickly becoming more expensive to sustain. Any donations are greatly appreciated, and 100% will go towards supporting Miza's future!
</p>
