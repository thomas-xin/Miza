if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		prog="server-dump",
		description="Discord Server Archiver",
	)
	parser.add_argument("-t", "--token", type=str, required=True)
	parser.add_argument("-g", "--guild-id", type=str, default="")
	parser.add_argument("-u", "--user-id", type=str, default="")
	parser.add_argument("-c", "--channel-id", type=str, default="")
	parser.add_argument("-l", "--log-id", type=str, default="")
	parser.add_argument("-m", "--min-id", type=int, default=None)
	parser.add_argument("-M", "--max-id", type=int, default=None)
	parser.add_argument("-p", "--filepath", type=str, required=True)
	parser.add_argument("-a", "--append", action=argparse.BooleanOptionalAction, default=True)
	ctx = parser.parse_args()

import os, json, time, asyncio, re, random, fractions, datetime, math, concurrent.futures
from traceback import print_exc
import niquests

from dynamic_dt import DynamicDT
from .types import pretty_json, lim_str, encode_jsonl, decode_jsonl
from .asyncs import gather, Semaphore
from .util import Request, temporary_file, unyt, url2ext, time_snowflake, snowflake_time_3, is_discord_message_link, split_message_link, parse_custom_ref, retrieve_api
from .caches import attachment_cache


main_sem = Semaphore(10, float("inf"), 1)
bucket_sems = {}
def get_sem(bucket=None):
	with main_sem:
		return bucket_sems.setdefault(bucket, Semaphore(5, float("inf"), 5))


headers = {}
async def verify_token(token):
	if token.startswith("~"):
		token = token.lstrip("~")
	else:
		base_token = token.rsplit(None, 1)[-1]
		if token.startswith("Bot "):
			resp = None
			base_token = token
		else:
			token = "Bot " + token
			headers.update({"Authorization": token, "Content-Type": "application/json", "Accept": "application/json"})
			resp = await Request.asession.get(
				"https://discord.com/api/v10/users/@me",
				headers=headers,
			)
		if not resp or resp.status_code == 401:
			token = base_token
			headers.update({"Authorization": token, "Content-Type": "application/json", "Accept": "application/json"})
			resp = await Request.asession.get(
				"https://discord.com/api/v10/users/@me",
				headers=headers,
			)
		resp.raise_for_status()
	if not token.startswith("Bot "):
		headers.update(Request.header())
	else:
		headers["User-Agent"] = "DiscordBot (https://mizabot.xyz, 1.0.0)"
	headers.update({"Authorization": token, "Content-Type": "application/json", "Accept": "application/json"})
	return headers

async def retrieve_api_a(path, method="GET", data=None):
	return await retrieve_api(path, method=method, headers=headers, data=data)


async def run(ctx):
	base_path = ctx.filepath
	os.makedirs(base_path, exist_ok=True)

	await verify_token(ctx.token)
	guild_ids = list(map(int, ctx.guild_id.split(","))) if ctx.guild_id else []
	channel_ids = list(map(int, ctx.channel_id.split(","))) if ctx.channel_id else []
	user_ids = list(map(int, ctx.user_id.split(","))) if ctx.user_id else []
	log_ids = set(map(int, ctx.log_id.split(","))) if ctx.log_id else []
	min_id, max_id = ctx.min_id, ctx.max_id
	if not min_id:
		min_id = 0
	if not max_id:
		max_id = time_snowflake(datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1))
	int_max = 18446744073709551615
	append = ctx.append

	progress: list[list[float]] = []
	def print_progress(mode="Progress"):
		cur, tot = sum(p[0] for p in progress), max(1, sum(p[1] for p in progress))
		print(f"\033[K\r{mode}: {'%.3f' % (cur / tot * 100)}%, {cur}/{tot}", end="")

	async def extract_channels(guild_ids, channel_ids):
		guild_ids = set(guild_ids)
		async with get_sem():
			resp = await retrieve_api_a(
				"users/@me/guilds",
			)
		servers = {int(data["id"]): data for data in resp}
		channels = {}

		async def get_channels_from(gid, index):
			async with get_sem(gid):
				resp = await retrieve_api_a(
					f"guilds/{gid}/channels",
				)
			channels.update({int(data["id"]): data for data in resp})
			async with get_sem(gid):
				resp = await retrieve_api_a(
					f"/guilds/{gid}/threads/active",
				)
			channels.update({int(data["id"]): data for data in resp["threads"]})

			progress[index][0] = 1
			print_progress("Guilds")

		async def get_channel(cid, index):
			async with get_sem(cid):
				resp = await retrieve_api_a(
					f"channels/{cid}",
				)
			channels[cid] = resp
			if "guild_id" in resp:
				guild_ids.add(int(resp["guild_id"]))

			progress[index][0] = 1
			print_progress("Channels")

		if channel_ids:
			progress.extend([[0, 1]] * len(channel_ids))
			print_progress("Channels")

			futs = [get_channel(cid, i) for i, cid in enumerate(channel_ids)]
			await gather(*futs)
			progress.clear()
		if guild_ids:
			progress.extend([[0, 1]] * len(guild_ids))
			print_progress("Guilds")

			futs = []
			for i, gid in enumerate(guild_ids):
				if gid not in servers:
					async with get_sem(gid):
						resp = await retrieve_api_a(
							f"guilds/{gid}",
						)
					servers[gid] = resp
				if not channel_ids:
					fut = get_channels_from(gid, i)
					futs.append(fut)
			await gather(*futs)
			progress.clear()
		return {gid: servers[gid] for gid in guild_ids}, channels

	servers, channels = await extract_channels(guild_ids, channel_ids)

	files = dict(
		sounds={},
		stickers={},
		emojis={},
		attachments={},
		users={},
	)

	ereg = re.compile("<a?:[A-Za-z0-9\\-~_]+:[0-9]+>")
	async def get_emojis(gid):
		async with get_sem(gid):
			resp = await retrieve_api_a(
				f"guilds/{gid}/emojis",
			)
		for e in resp:
			eid = e["id"]
			url = f"https://cdn.discordapp.com/emojis/{eid}.webp"
			if e.get("animated"):
				url += "?animated=true"
			files["emojis"][eid] = [e["name"] + ".webp", url]

		async with get_sem(gid):
			resp = await retrieve_api_a(
				f"guilds/{gid}/stickers",
			)
		for s in resp:
			sid = s["id"]
			if s.get("format_type") == 3:
				url = f"https://discord.com/stickers/{sid}.json"
			else:
				url = f"https://media.discordapp.net/stickers/{sid}.png"
			files["stickers"][sid] = [s["name"] + f".{url2ext(url)}", url]

		async with get_sem(gid):
			resp = await retrieve_api_a(
				f"guilds/{gid}/soundboard-sounds",
			)
		for s in resp.get("items", ()):
			sid = s["sound_id"]
			url = f"https://cdn.discordapp.com/soundboard-sounds/{sid}"
			files["sounds"][sid] = [s["name"] + ".ogg", url]

	futs = []
	for gid in servers:
		fut = asyncio.create_task(get_emojis(gid))
		futs.append(fut)
		server = servers[gid]
		s = pretty_json(server)
		cur_path = f"{base_path}/guilds/{gid}"
		os.makedirs(cur_path, exist_ok=True)
		with open(f"{cur_path}/info.json", "w", encoding="utf-8") as f:
			f.write(s)
	await gather(*futs)

	message_map = {}
	channel_map = {}

	def process_miza_log(m, e):
		desc = e["description"]
		content, url = desc.rsplit("\n\n", 1)
		reacts, url = url.strip().split("[View Message]", 1)
		if not is_discord_message_link(url):
			raise StopIteration
		content = content.split("\n", 1)[1].strip()
		author = e.get("author", {})
		avatar_url = author.get("icon_url")
		try:
			a_url = author["url"]
			assert a_url.startswith("https://cdn.discordapp.com/avatars/")
			author_id = a_url.removeprefix("https://cdn.discordapp.com/avatars/").split("/", 1)[0]
			avatar_url = None
		except (KeyError, AssertionError):
			author_id = desc.split(None, 1)[0].strip("<@!> ")
		_gid, _cid, _mid = split_message_link(url.removesuffix(")"))
		m2 = dict(
			type=-1,
			content=content,
			id=_mid,
			channel_id=_cid,
			mentions=[],
			attachments=[],
			timestamp=DynamicDT.fromdatetime(snowflake_time_3(int(_mid))).as_iso(),
			edited_timestamp=None,
			flags=0,
			components=[],
			author=dict(
				id=author_id,
				username=author.get("name", "Unknown User"),
				avatar=author["icon_url"].rsplit("/", 1)[-1].split(".", 1)[0] if author.get("icon_url") else "0",
				avatar_url=avatar_url,
				discriminator="0",
				public_flags=65536,
				flags=65536,
				bot=False,
			),
			pinned=False,
			mention_everyone=False,
			tts=False,
			embeds=[],
			reactions=[],
		)
		if not avatar_url and author.get("icon_url") and (author_id := int(author_id)):
			url = author["icon_url"]
			files["users"][author_id] = [str(m2["author"]["username"]) + f".{url2ext(url)}", url]
		for field in e.get("fields", ()):
			e2 = dict(
				type="rich",
				title=field.get("name"),
				description=field.get("value", ""),
			)
			m2["embeds"].append(e2)
		if e.get("image") or e.get("thumbnail"):
			if not m2["embeds"]:
				m2["embeds"].append(dict(
					type="rich",
				))
			m2["embeds"][0].update(dict(
				image=e.get("image"),
				thumbnail=e.get("thumbnail"),
			))
		for e in m["embeds"][1:]:
			if not e.get("image") and not e.get("thumbnail"):
				continue
			e2 = dict(
				type="rich",
				image=e.get("image"),
				thumbnail=e.get("thumbnail"),
			)
			m2["embeds"].append(e2)
		reacts = reacts.strip().split()
		for k, v in zip(reacts[::2], reacts[1::2]):
			if not ereg.fullmatch(k):
				r = dict(
					emoji=dict(
						id=None,
						name=k,
					),
					count=v,
				)
			else:
				er = k
				try:
					eid = int(er.removesuffix(">").rsplit(":", 1)[-1])
				except:
					print_exc()
				else:
					animated = er.startswith("<a:")
					name = er.split(":", 2)[1]
					url = f"https://cdn.discordapp.com/emojis/{eid}.webp"
					if animated:
						url += "?animated=true"
					files["emojis"].setdefault(eid, [name + ".webp", url])
				r = dict(
					emoji=dict(
						id=str(eid),
						name=name,
					),
					count=v,
				)
			m2["reactions"].append(r)
		return m2

	def process_carlbot_log(m, e):
		channel_name = e["title"].split(" in ", 1)[-1].lstrip("#")
		if channel_name in ambiguous_channel_names:
			raise StopIteration
		try:
			_cid = channels_by_name[channel_name]
		except KeyError:
			raise StopIteration
		_mid = e["description"].rsplit("\n\n", 1)[-1].split(": ", 1)[-1].strip()
		if not _mid.isnumeric():
			raise StopIteration
		author = e.get("author", {})
		avatar_url = author.get("icon_url")
		try:
			a_url = author["url"]
			assert a_url.startswith("https://cdn.discordapp.com/avatars/")
			author_id = a_url.removeprefix("https://cdn.discordapp.com/avatars/").split("/", 1)[0]
			avatar_url = None
		except (KeyError, AssertionError):
			author_id = e.get("footer", {}).get("text", "").split(": ", 1)[-1].strip() or "0"
		m2 = dict(
			type=-1,
			content=e["description"].rsplit("\n\n", 1)[0].strip(),
			id=_mid,
			channel_id=_cid,
			mentions=[],
			attachments=[],
			timestamp=DynamicDT.fromdatetime(snowflake_time_3(int(_mid))).as_iso(),
			edited_timestamp=None,
			flags=0,
			components=[],
			author=dict(
				id=author_id,
				username=author.get("name", "Unknown User"),
				avatar=author["icon_url"].rsplit("/", 1)[-1].split(".", 1)[0] if author.get("icon_url") else "0",
				avatar_url=avatar_url,
				discriminator="0",
				public_flags=65536,
				flags=65536,
				bot=False,
			),
			pinned=False,
			mention_everyone=False,
			tts=False,
			embeds=[],
			reactions=[],
		)
		if not avatar_url and author.get("icon_url") and (author_id := int(author_id)):
			url = author["icon_url"]
			files["users"][author_id] = [str(m2["author"]["username"]) + f".{url2ext(url)}", url]
		return m2

	def process_message(m, cid):
		cur_cid = m.get("channel_id", cid)
		for a in m.get("attachments", ()):
			attachment_cache.store(a["url"])
			url = unyt(a["url"])
			files["attachments"][a["id"]] = [a["filename"], url]
		cur_id = int(m["id"])
		for s in m.get("sticker_items", ()):
			sid = s["id"]
			if s.get("format_type") == 3:
				url = f"https://discord.com/stickers/{sid}.json"
			else:
				url = f"https://media.discordapp.net/stickers/{sid}.png"
			files["stickers"][sid] = [s["name"] + "." + url.rsplit(".", 1)[-1], url]
		author = m.get("author", {})
		if author.get("bot") and int(cur_cid) in log_ids and (embeds := m.get("embeds")):
			try:
				e = embeds[0]
				if e.get("type") != "rich":
					raise StopIteration
				desc = e.get("description", "")
				if e.get("title", "").startswith("Message deleted in "):
					m2 = process_carlbot_log(m, e)
				elif " from**" in desc and "[View Message]" in desc:
					m2 = process_miza_log(m, e)
				elif desc.endswith(" has been updated:") and e.get("thumbnail") and any(field["name"] == "Avatar" for field in e.get("fields", ())) and e.get("author"):
					aid = int(desc.split(None, 1)[0].strip("<@!>"))
					url = e["thumbnail"]["url"]
					files["users"][aid] = [str(e["author"].get("name", "Unknown User")) + f".{url2ext(url)}", url]
					m2 = m
				else:
					raise StopIteration # TODO: Implement other popular Discord bot message log formats
				process_message(m2, m2.get("channel_id", cid))
			except StopIteration:
				pass
		elif author.get("avatar") and not author.get("bot"):
			ext = "gif" if author["avatar"].startswith("a_") else "png"
			url = f"https://cdn.discordapp.com/avatars/{author['id']}/{author['avatar']}.{ext}"
			aid = int(author["id"])
			if aid not in files["users"]:
				files["users"][aid] = [str(author["username"]) + f".{url2ext(url)}", url]
		for r in m.get("reactions", ()):
			e = r.get("emoji")
			if not e or not e.get("id"):
				continue
			eid = e["id"]
			url = f"https://cdn.discordapp.com/emojis/{eid}.webp"
			if e.get("animated"):
				url += "?animated=true"
			files["emojis"][eid] = [e["name"] + ".webp", url]
		found = ereg.findall(m.get("content", ""))
		for er in found:
			try:
				eid = int(er.removesuffix(">").rsplit(":", 1)[-1])
			except:
				print_exc()
			else:
				animated = er.startswith("<a:")
				name = er.split(":", 2)[1]
				url = f"https://cdn.discordapp.com/emojis/{eid}.webp"
				if animated:
					url += "?animated=true"
				files["emojis"].setdefault(eid, [name + ".webp", url])
		info = parse_custom_ref(m.get("content", ""))
		if info:
			m.update(info)
		message_map.setdefault(cur_cid, {})[cur_id] = m
		return cur_id

	async def scan_channel(channel, index):
		gid = channel.get("guild_id", 0)
		cid = channel["id"]
		s = pretty_json(channel)
		if gid:
			if channel.get("type") in (10, 11, 12) and (pid := channel.get("parent_id")):
				path = f"guilds/{gid}/channels/{pid}/threads/{cid}"
			else:
				path = f"guilds/{gid}/channels/{cid}"
		else:
			path = f"channels/{cid}"
		cur_path = base_path + "/" + path
		channel_map[cid] = cur_path
		os.makedirs(cur_path, exist_ok=True)
		# print(cid, f"{cur_path}/info.json")
		with open(f"{cur_path}/info.json", "w", encoding="utf-8") as f:
			f.write(s)
		mid, Mid = min_id, max_id
		messages = message_map.setdefault(cid, {})

		try:
			n = 1
			if user_ids and gid:
				while True:
					last_id = mid
					offset = 0
					while offset < 9975:
						path = f"guilds/{gid}/messages/search"
						query = f"?include_nsfw=true&channel_id={cid}&limit=25&offset={offset}&max_id={Mid}&min_id={mid}&sort_by=timestamp&sort_order=asc"
						query += "".join(f"&author_id={u}" for u in user_ids)
						async with get_sem(gid):
							resp = await retrieve_api_a(f"{path}{query}")
						if not resp.get("messages"):
							raise StopIteration
						for m in resp["messages"]:
							if isinstance(m, list):
								m = m[0]
							cur_id = process_message(m, cid)
							last_id = max(last_id, cur_id)
						offset += len(resp["messages"])
						if offset >= resp["total_results"]:
							raise StopIteration
						progress[index][1] = resp["total_results"]
						progress[index][0] += len(resp["messages"])
						print_progress("Messages")
						await asyncio.sleep(len(progress) / n)
						if not any(p[1] == int_max for p in progress):
							n += 1
					mid = max(last_id, mid)
			else:
				first = bool(gid)
				while True:
					async with get_sem(cid):
						if first:
							path = f"guilds/{gid}/messages/search"
							query = f"?include_nsfw=true&channel_id={cid}&limit=25&offset=0&max_id={Mid}&min_id={mid}&sort_by=timestamp&sort_order=asc"
							resp = await retrieve_api_a(f"{path}{query}")
							first = False
							progress[index][1] = resp["total_results"]
							resp = resp["messages"]
						else:
							resp = await retrieve_api_a(
								f"channels/{cid}/messages?limit=100&after={mid}",
							)
					if not resp:
						raise StopIteration
					assert isinstance(resp, list), str(resp)
					for m in resp:
						if isinstance(m, list):
							m = m[0]
						if user_ids and m.get("author", {}).get("id", 0) not in user_ids:
							continue
						cur_id = process_message(m, cid)
						if cur_id > Mid:
							messages.pop(cur_id, None)
							raise StopIteration
						mid = max(mid, cur_id)
					if not gid:
						progress[index][1] = max_id - min_id
						progress[index][0] = mid - min_id
					else:
						progress[index][0] += len(resp)
					print_progress("Messages")
					await asyncio.sleep(len(progress) / n)
					if not any(p[1] == int_max for p in progress):
						n += 1
		except StopIteration:
			pass
		except niquests.exceptions.HTTPError:
			pass
		except Exception:
			print_exc()
		progress[index][0] = math.ceil(0.99 * progress[index][1])

	channels_by_name = {}
	ambiguous_channel_names = set()
	futs = []
	futs2 = []

	async def scan_threads(cid):
		for mode in ("public", "private"):
			async with get_sem(cid):
				try:
					resp = await retrieve_api_a(
						f"channels/{cid}/threads/archived/{mode}",
					)
				except niquests.exceptions.HTTPError:
					break
			for thread in resp.get("threads", ()):
				tid = thread["id"]
				name = thread.get("name", "")
				if name in channels_by_name:
					ambiguous_channel_names.add(name)
				else:
					channels_by_name[name] = tid
				fut = scan_channel(
					thread,
					len(progress),
				)
				progress.append([0, int_max])
				if tid in log_ids:
					futs2.append(fut)
				else:
					await fut

	for cid, channel in channels.items():
		if channel["type"] not in (0, 1, 2, 3, 5, 10, 11, 12, 15):
			continue
		name = channel.get("name", "")
		if name in channels_by_name:
			ambiguous_channel_names.add(name)
		else:
			channels_by_name[name] = cid
		if channel["type"] not in (1, 2, 3, 10, 11, 12, 15):
			fut = scan_threads(
				cid,
			)
			futs.append(fut)
		fut = scan_channel(
			channel,
			len(progress),
		)
		progress.append([0, int_max])
		if int(cid) in log_ids:
			futs2.append(fut)
		else:
			futs.append(asyncio.create_task(fut))
	while futs:
		await futs.pop(0)
	await gather(*futs2)
	for index, (cid, messages) in enumerate(message_map.items()):
		try:
			cur_path = channel_map[cid]
		except KeyError:
			continue
		target = f"{cur_path}/messages.jsonl"
		if append and os.path.exists(target) and os.path.getsize(target):
			with open(target, "rb") as f:
				b = f.read()
			mlist2 = decode_jsonl(b)
			ms2 = {int(m["id"]): m for m in mlist2}
			ms2.update(messages)
			messages = ms2
		b = encode_jsonl(messages[i] for i in sorted(messages))
		with open(target, "wb") as f:
			f.write(b)
		if index in range(len(progress)):
			progress[index][0] = progress[index][1]
		print_progress("Messages")
	progress.clear()

	dm_path = f"{base_path}/guilds/0"
	if os.path.exists(dm_path):
		dms = dict(
			id=0,
			name="Direct Messages",
			icon_url="https://cdn.discordapp.com/embed/avatars/0.png",
			icon=None,
		)
		s = pretty_json(dms)
		with open(f"{dm_path}/info.json", "w") as f:
			f.write(s)

	progress.append([0, 1])
	futs = []

	def load_file_type(kind, data):
		base_fold = f"{base_path}/{kind}"
		if os.path.exists(base_fold):
			t = time.time()
			os.utime(base_fold, (t, t))
		invalid_pattern = r'[<>:"/\\|?*\x00-\x1F]'
		replacement = "_"

		async def download_single(url, fn):
			try:
				await attachment_cache.download(url, filename=fn)
			except Exception:
				print_exc()
				return
			progress[0][0] += 1
			print_progress("Files")

		for oid, (name, url) in data.items():
			path = f"{base_fold}/{oid}"
			os.makedirs(path, exist_ok=True)
			name = lim_str(re.sub(invalid_pattern, replacement, name).strip(), 128)
			fn = f"{path}/{name}"
			if append and os.path.exists(fn) and os.path.getsize(fn):
				continue
			fut = download_single(url, fn)
			yield fut

	for kind, data in files.items():
		futs.extend(load_file_type(kind, data))
	random.shuffle(futs)
	progress[0][1] = len(futs)
	await gather(*futs, return_exceptions=True, max_concurrency=32)
	progress[0][0] = progress[0][1]
	print_progress("Files")


if __name__ == "__main__":
	eloop = asyncio.get_event_loop()
	eloop.run_until_complete(run(ctx))