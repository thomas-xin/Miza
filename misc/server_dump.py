import sys

if len(sys.argv) < 9:
	print(f"Usage: python {sys.argv[0]} <token> <guild-id> <user-ids> <channel-ids> <logs-to-parse> <min-id> <max-id> <filepath>")
	sys.exit(1)

import os, json, time, asyncio, re, random, datetime, concurrent.futures
from traceback import print_exc
import niquests

from dynamic_dt import DynamicDT
from .types import pretty_json, lim_str
from .asyncs import run_async, gather, Semaphore
from .util import Request, temporary_file, unyt, time_snowflake, snowflake_time_3, is_discord_message_link
from .caches import attachment_cache


base_path = " ".join(sys.argv[8:])
os.makedirs(base_path, exist_ok=True)

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

async def retrieve_api(path, method="GET", data=None):
	url = f"https://discord.com/api/v10/{path}"
	for attempt in range(16):
		delay = (1 << attempt) * (random.random() + 1)
		try:
			resp = await Request.asession.request(
				method,
				url,
				data=data,
				headers=headers,
			)
		except (
			niquests.ConnectionError,
			niquests.ConnectTimeout,
			niquests.ReadTimeout,
			niquests.Timeout,
			niquests.exceptions.ChunkedEncodingError,
		):
			await asyncio.sleep(delay)
			continue
		if resp.status_code in (202, 429, 502, 503):
			try:
				msg = resp.json()
			except Exception:
				await asyncio.sleep(delay)
			else:
				if isinstance(msg, dict) and "message" in msg and "retry_after" in msg:
					await asyncio.sleep(float(msg["retry_after"]) + delay / 6)
				else:
					await asyncio.sleep(delay)
		else:
			try:
				resp.raise_for_status()
				return resp.json()
			except Exception as ex:
				print(resp, url, repr(ex), headers, resp.text)
				raise
	raise RuntimeError("Maximum request attempts exceeded.")


async def run():
	await verify_token(sys.argv[1])
	guild_ids = list(map(int, sys.argv[2].split(","))) if sys.argv[2] else []
	user_ids = list(map(int, sys.argv[3].split(","))) if sys.argv[3] else []
	channel_ids = list(map(int, sys.argv[4].split(","))) if sys.argv[4] else []
	log_ids = set(map(int, sys.argv[5].split(","))) if sys.argv[5] else []
	min_id, max_id = sorted(map(int, sys.argv[6:8]))
	if not max_id:
		max_id = time_snowflake(datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1))

	progress: list[float] = []
	def print_progress(mode="Progress"):
		print(f"{mode}: {'%.3f' % (sum(progress) / len(progress) * 100)}%", end="\r")

	async def extract_channels(guild_ids, channel_ids):
		guild_ids = set(guild_ids)
		async with get_sem():
			resp = await retrieve_api(
				"users/@me/guilds",
			)
		servers = {int(data["id"]): data for data in resp}
		channels = {}

		async def get_channels_from(gid, index):
			async with get_sem(gid):
				resp = await retrieve_api(
					f"guilds/{gid}/channels",
				)
			channels.update({int(data["id"]): data for data in resp})

			progress[index] = 1
			print_progress("Guilds")

		async def get_channel(cid, index):
			async with get_sem(cid):
				resp = await retrieve_api(
					f"channels/{cid}",
				)
			channels[cid] = resp
			guild_ids.add(int(resp["guild_id"]))

			progress[index] = 1
			print_progress("Channels")

		if channel_ids:
			progress.extend([0] * len(channel_ids))
			print_progress("Channels")

			futs = [get_channel(cid, i) for i, cid in enumerate(channel_ids)]
			await gather(*futs)
			progress.clear()
		if guild_ids:
			progress.extend([0] * len(guild_ids))
			print_progress("Guilds")

			futs = []
			for i, gid in enumerate(guild_ids):
				if gid not in servers:
					async with get_sem(gid):
						resp = await retrieve_api(
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
	)

	ereg = re.compile("<a?:[A-Za-z0-9\\-~_]+:[0-9]+>")
	async def get_emojis(gid):
		async with get_sem(gid):
			resp = await retrieve_api(
				f"guilds/{gid}/emojis",
			)
		for e in resp:
			eid = e["id"]
			url = f"https://cdn.discordapp.com/emojis/{eid}.webp"
			if e.get("animated"):
				url += "?animated=true"
			files["emojis"][eid] = [e["name"] + ".webp", url]

		async with get_sem(gid):
			resp = await retrieve_api(
				f"guilds/{gid}/stickers",
			)
		for s in resp:
			sid = s["id"]
			if s.get("format_type") == 3:
				url = f"https://discord.com/stickers/{sid}.json"
			else:
				url = f"https://media.discordapp.net/stickers/{sid}.png"
			files["stickers"][sid] = [s["name"] + "." + url.rsplit(".", 1)[-1], url]

		async with get_sem(gid):
			resp = await retrieve_api(
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
		try:
			if int(cur_cid) in log_ids:
				embeds = m.get("embeds", ())
				if not embeds:
					raise StopIteration
				e = embeds[0]
				if e.get("type") != "rich":
					raise StopIteration
				desc = e.get("description", "")
				if "**deleted message from**" not in desc or "[View Message]" not in desc:
					raise StopIteration
				content, url = desc.rsplit("\n\n", 1)
				content = content.split("\n", 1)[1].strip()
				author_id = desc.split(None, 1)[0].strip("<@!> ")
				reacts, url = url.strip().split("[View Message]", 1)
				if not is_discord_message_link(url):
					raise StopIteration
				*_, _gid, _cid, _mid = url.rstrip(")").rsplit("/", 3)
				author = e.get("author", {})
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
						username=author.get("name"),
						avatar=author["url"].rsplit("/", 1)[-1].split(".", 1)[0] if author.get("url") else "0",
						avatar_url=author["url"],
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
				for e in embeds[1:]:
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
				process_message(m2, _cid)
		except StopIteration:
			pass
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
		message_map.setdefault(cur_cid, {})[cur_id] = m
		return cur_id

	async def scan_channel(channel, path, index):
		gid = channel["guild_id"]
		cid = channel["id"]
		s = pretty_json(channel)
		cur_path = base_path + "/" + path
		channel_map[cid] = cur_path
		os.makedirs(cur_path, exist_ok=True)
		with open(f"{cur_path}/info.json", "w", encoding="utf-8") as f:
			f.write(s)
		mid, Mid = min_id, max_id
		messages = message_map.setdefault(cid, {})

		try:
			if user_ids:
				while True:
					last_id = mid
					offset = 0
					while offset < 9975:
						path = f"guilds/{gid}/messages/search"
						query = f"?include_nsfw=true&channel_id={cid}&limit=25&offset={offset}&max_id={Mid}&min_id={mid}&sort_by=timestamp&sort_order=asc"
						query += "".join(f"&author_id={u}" for u in user_ids)
						async with get_sem(gid):
							resp = await retrieve_api(f"{path}{query}")
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
						progress[index] = (last_id - min_id) / (max_id - min_id)
						print_progress("Messages")
					mid = max(last_id, mid)
			else:
				n = 0
				while True:
					if not n & 3 and sum(progress) / len(progress) >= 0.95:
						path = f"guilds/{gid}/messages/search"
						query = f"?include_nsfw=true&channel_id={cid}&limit=25&max_id={Mid}&min_id={mid}&sort_by=timestamp&sort_order=asc"
						async with get_sem(gid):
							resp = await retrieve_api(f"{path}{query}")
						resp = resp["messages"]
					else:
						async with get_sem(cid):
							resp = await retrieve_api(
								f"channels/{cid}/messages?limit=100&after={mid}",
							)
					if not resp:
						raise StopIteration
					assert isinstance(resp, list), str(resp)
					for m in resp:
						if isinstance(m, list):
							m = m[0]
						cur_id = process_message(m, cid)
						if cur_id > Mid:
							messages.pop(cur_id, None)
							raise StopIteration
						mid = max(mid, cur_id)
					progress[index] = (mid - min_id) / (max_id - min_id)
					print_progress("Messages")
					n += 1
		except StopIteration:
			pass
		except niquests.exceptions.HTTPError:
			pass
		progress[index] = 0.99

	async def scan_threads(cid):
		for mode in ("public", "private"):
			async with get_sem(cid):
				try:
					resp = await retrieve_api(
						f"channels/{cid}/threads/archived/{mode}",
					)
				except niquests.exceptions.HTTPError:
					break
			for thread in resp.get("threads", ()):
				progress.append(0)
				await scan_channel(
					thread,
					f"guilds/{gid}/channels/{cid}/threads/{thread['id']}",
					len(progress) - 1,
				)

	futs = []
	for cid, channel in channels.items():
		if channel["type"] not in (0, 1, 2, 3, 5, 10, 11, 12, 15):
			continue
		gid = int(channel["guild_id"])
		if channel["type"] not in (2, 10, 11, 12, 15):
			fut = scan_threads(
				cid,
			)
			futs.append(fut)
		fut = scan_channel(
			channel,
			f"guilds/{gid}/channels/{cid}",
			len(progress),
		)
		progress.append(0)
		futs.append(fut)
	await gather(*futs)
	for index, (cid, messages) in enumerate(message_map.items()):
		try:
			cur_path = channel_map[cid]
		except KeyError:
			continue
		s = pretty_json([messages[i] for i in sorted(messages)])
		with open(f"{cur_path}/messages.json", "w", encoding="utf-8") as f:
			f.write(s)
		progress[index] = 1
		print_progress("Messages")
	progress.clear()

	async def load_file_type(kind, data, index):
		sem = Semaphore(10, float("inf"), 1)
		invalid_pattern = r'[<>:"/\\|?*\x00-\x1F]'
		replacement = "_"
		c = 0
		fut = None
		for oid, (name, url) in data.items():
			path = f"{base_path}/{kind}/{oid}"
			os.makedirs(path, exist_ok=True)
			name = lim_str(re.sub(invalid_pattern, replacement, name), 128)
			fn = f"{path}/{name}"
			if os.path.exists(fn) and os.path.getsize(fn):
				continue
			async with sem:
				fut2 = asyncio.create_task(attachment_cache.download(url, filename=fn))
				if fut:
					try:
						await fut
					except Exception:
						print_exc()
				fut = fut2
			c += 1
			progress[index] = c / len(data)
			print_progress("Files")
		if fut:
			try:
				await fut
			except Exception:
				print_exc()
		progress[index] = 1
		print_progress("Files")

	progress.extend([0] * len(files))
	futs = []
	for index, (kind, data) in enumerate(files.items()):
		fut = load_file_type(kind, data, index)
		futs.append(fut)
	await gather(*futs)
	progress = [1]
	print_progress("Files")


eloop = asyncio.get_event_loop()
eloop.run_until_complete(run())