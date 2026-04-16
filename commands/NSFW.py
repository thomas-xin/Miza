# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT


class Verify(Command):
	name = ["AgeVerify"]
	min_level = 0
	description = "Verifies your account age as 18+, allowing you to access NSFW-restricted commands within DM channels."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("enable", "disable", "view"),
			),
			description="Determines whether to enable, disable, or view verification status",
			example="enable",
			default="view",
		),
	)
	rate_limit = (1, 6)

	async def __call__(self, bot, _guild, _user, _nsfw, _name, mode, **void):
		nsfw = bot.get_userbase(_user.id, "nsfw", None)
		if not _nsfw and nsfw is None:
			raise PermissionError(f"This command is only available in {uni_str('NSFW')} channels, or for users who have posted in at least one {uni_str('NSFW')} channel shared with {bot.name}.")
		if mode == "disable":
			bot.set_userbase(_user.id, "nsfw", False)
			return italics(css_md(f"Disabled age-verified DMs for {sqr_md(_user)}."))
		elif mode == "enable":
			bot.set_userbase(_user.id, "nsfw", True)
			return italics(css_md(f"Enabled age-verified DMs for {sqr_md(_user)}."))
		if not nsfw:
			return ini_md(f'Age-verified DMs are currently disabled for {sqr_md(_user)}. Use "{bot.get_prefix(_guild)}{_name} enable" to enable.')
		return ini_md(f"Age-verified DMs are currently enabled for {sqr_md(_user)}.")


class UpdateNSFW(Database):
	name = "nsfw"
	no_file = True

	def _send_(self, message, **void):
		channel = message.channel
		if is_nsfw(channel):
			user = message.author
			if user.id not in self:
				self[user.id] = False