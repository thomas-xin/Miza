import sys, os

if len(sys.argv) < 3:
	print(f"Usage: python {sys.argv[0]} <token> <server-id>")
	sys.exit(1)

compressible = {"txt", "exe", "py", "jar", "js", "c", "pl", "json", "php", "svg", "html", "css", "xml", "md", "pdf", "doc", "xls", "ppt", "docx", "xlsx", "pptx", "eot", "swf"}
mime_map = {
	"plain": "txt",
	"mpeg": "mp3",
	"octet-stream": "bin",
	"jpeg": "jpg",
	"msdos-program": "exe",
	"python": "py",
	"java-archive": "jar",
	"javascript": "js",
	"httpd-php": "php",
	"quicktime": "qt",
	"svg+xml": "svg",
	"xhtml+xml": "html",
	"msword": "doc",
	"excel": "xls",
	"powerpoint": "ppt",
	"vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
	"vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
	"vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
	"bzip": "bz",
	"bzip2": "bz2",
	"vnd.ms-fontobject": "eot",
	"epub+zip": "epub",
	"vnd.rar": "rar",
	"shockwave-flash": "swf",
	"mp2t": "ts",
	"vnd.visio": "vsd",
	"7z-compressed": "7z",
	"msvideo": "avi",
}

def map_mime(fmt):
	fmt = fmt.split(";", 1)[0].split("/", 1)[-1].removeprefix("x-").removeprefix("ms-")
	return mime_map.get(fmt, fmt)

PROGRESS = 0

if sys.argv[1] == "-map_mime":
	folder = " ".join(sys.argv[2:])
	for x, y, z in os.walk(folder):
		for fn in z:
			try:
				name, ctype = fn.rsplit(".", 1)
			except ValueError:
				continue
			fn2 = name + "." + map_mime(ctype)
			if fn2 != fn:
				os.rename(x + "/" + fn, x + "/" + fn2)
				PROGRESS += 1
	print(f"Progress: {PROGRESS} (Complete)", end="")
	sys.exit(0)

import requests, json, time, concurrent.futures, random, re
from traceback import print_exc
reqs = requests.Session()
token, gid = sys.argv[1], sys.argv[2]

exc = concurrent.futures.ThreadPoolExecutor(max_workers=64)
futs = []
PTIME = 0

if token.startswith("~"):
	token = token.lstrip("~")
else:
	if token.startswith("Bot "):
		resp = None
	else:
		headers = {"Authorization": "Bot " + token, "Content-Type": "application/json"}
		resp = reqs.get(
			"https://discord.com/api/v10/users/@me",
			headers=headers,
		)
		PROGRESS += 1
	if not resp or resp.status_code == 401:
		headers = {"Authorization": token, "Content-Type": "application/json"}
		resp = reqs.get(
			"https://discord.com/api/v10/users/@me",
			headers=headers,
		)
		PROGRESS += 1
	else:
		token = "Bot " + token
	resp.raise_for_status()
headers = {"Authorization": token, "Content-Type": "application/json"}

if "," in gid:
	if "(" not in gid and "[" not in gid and "{" not in gid:
		gid = f"[{gid}]"
	import ast
	cids = ast.literal_eval(gid)
	server = [{}]
	channels = []
	for cid in cids:
		resp = reqs.get(
			f"https://discord.com/api/v10/channels/{cid}",
			headers=headers,
		)
		PROGRESS += 1
		resp.raise_for_status()
		channel = resp.json()
		channels.append(channel)
		if not server[0].get("name"):
			server[0]["name"] = channel.get("name")
			if len(cids) > 1:
				server[0]["name"] += f" +{len(cids) - 1}"
	gid = None
else:
	resp = reqs.get(
		"https://discord.com/api/v10/users/@me/guilds",
		headers=headers,
	)
	PROGRESS += 1
	resp.raise_for_status()
	d = resp.json()

	server = [s for s in d if int(gid) == int(s["id"])]
	if not server:
		raise FileNotFoundError("Server not found.")

	resp = reqs.get(
		f"https://discord.com/api/v10/guilds/{gid}/channels",
		headers=headers,
	)
	PROGRESS += 1
	channels = resp.json()

fn = " ".join(sys.argv[3:])
if fn:
	sfold = str(time.time_ns() // 1000)
	END = "\n"
else:
	sfold = server[0]["name"]
	END = ""
if not sfold.endswith("/"):
	sfold += "/"

if not os.path.exists(sfold):
	os.mkdir(sfold)

def header():
	return {
		"User-Agent": f"Mozilla/5.{random.randint(1, 9)}",
		"DNT": "1",
		"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
	}

def proxy_download(url, fn=None, proxy=True, timeout=24):
	if proxy:
		loc = random.choice(("eu", "us"))
		i = random.randint(1, 17)
		stream = f"https://{loc}{i}.proxysite.com/includes/process.php?action=update"
		# print(url, stream, fn, sep="\n")
		req = reqs.post(
			stream,
			data=dict(d=url, allowCookies="on"),
			headers=header(),
			timeout=timeout,
			stream=True,
		)
	else:
		req = reqs.get(
			url,
			headers=header(),
			timeout=timeout,
			stream=True
		)
	with req as resp:
		if resp.status_code not in range(200, 400):
			raise ConnectionError(resp.status_code, resp)
		if not fn:
			resp.content
			return resp
		try:
			size = int(resp.headers["Content-Length"])
		except (KeyError, ValueError):
			size = None
		it = resp.iter_content(65536)
		with open(fn, "wb") as f:
			try:
				while True:
					b = next(it)
					if not b:
						break
					f.write(b)
			except StopIteration:
				pass
			except:
				print_exc()
		if size:
			for i in range(3):
				pos = os.path.getsize(fn)
				if pos >= size:
					break
				# print(f"Incomplete download ({pos} < {size}), resuming...")
				h = header()
				h["Range"] = f"bytes={pos}-"
				resp = reqs.get(url, headers=h, timeout=timeout, stream=True)
				resp.raise_for_status()
				it = resp.iter_content(65536)
				with open(fn, "ab") as f:
					try:
						while True:
							b = next(it)
							if not b:
								raise StopIteration
							f.write(b)
					except StopIteration:
						continue
					except:
						print_exc()
			pos = os.path.getsize(fn)
			if pos < size:
				raise EOFError(f"{url}: Incomplete download ({pos} < {size}), unable to resolve.")
		return fn

def download(url, fn, backup=None):
	try:
		resp = proxy_download(url, proxy=False)
	except:
		print_exc()
		resp = reqs.get(backup or url)
	globals()["PROGRESS"] += 1
	if resp.headers.get("Content-Type"):
		ctype = resp.headers["Content-Type"]
		fn += "." + map_mime(ctype)
	with open(fn, "wb") as f:
		f.write(resp.content)
	t = time.time()
	if t - PTIME >= 1:
		globals()["PTIME"] = t
		sys.stdout.write(f"\rProgress: {PROGRESS}{END}")
		sys.stdout.flush()

if gid:
	resp = reqs.get(
		f"https://discord.com/api/v10/guilds/{gid}/emojis",
		headers=headers,
	)
	emojis = {int(e["id"]): e for e in resp.json()}
else:
	emojis = {}

ereg = re.compile("<a?:[A-Za-z0-9\\-~_]+:[0-9]+>")
users = {}
while channels:
	channel = channels.pop(0)
	if channel["type"] not in (0, 1, 3, 5, 10, 11, 12, 15):
		continue
	cid = channel["id"]
	for mode in ("public", "private"):
		resp = reqs.get(
			f"https://discord.com/api/v10/channels/{cid}/threads/archived/{mode}",
			headers=headers,
		)
		PROGRESS += 1
		try:
			d = resp.json()["threads"]
		except KeyError:
			continue
		channels.extend(d)
	cname = channel["name"]
	cfold = cname + " ~ " + cid + "/"
	if not os.path.exists(sfold + cfold):
		os.mkdir(sfold + cfold)
	with open(sfold + cfold + cname + ".txt", "wb") as f:
		MID = 0
		data = []
		while True:
			resp = reqs.get(
				f"https://discord.com/api/v10/channels/{cid}/messages?limit=100&after={MID}",
				headers=headers,
			)
			PROGRESS += 1
			d = resp.json()
			data.extend(d)
			for m in sorted(d, key=lambda m: int(m["id"])):
				mid = int(m["id"])
				MID = max(mid, MID)
				a = m["author"]
				users[a["id"]] = a
				s = a["username"]
				if a.get("bot"):
					s += " [BOT]"
				s += " at " + m["timestamp"].rsplit("+", 1)[0] + "\n"
				s += m["content"]
				for i, a in enumerate(m["attachments"]):
					aid = a["id"]
					if i or m["content"]:
						s += "\n"
					s += f"<attachment {aid}>"
					if not os.path.exists(f"{sfold}{cfold}attachments"):
						os.mkdir(f"{sfold}{cfold}attachments")
					futs.append(exc.submit(download, a["url"], f"{sfold}{cfold}attachments/{aid}"))
				for i, si in enumerate(m.get("sticker_items", ())):
					sid = si["id"]
					if si.get("format_type") == 3:
						url = f"https://discord.com/stickers/{sid}.json"
					else:
						url = f"https://media.discordapp.net/stickers/{sid}.png"
					if i or m["content"] or m["attachments"]:
						s += "\n"
					s += f"<{url}>"
				for i, e in enumerate(m["embeds"]):
					if i or m["content"] or m["attachments"] or m.get("sticker_items"):
						s += "\n"
					s += "{\n"
					title = e.get("title")
					if title:
						s += title + "\n"
					description = e.get("description")
					if description:
						s += description + "\n"
					image = e.get("image")
					if image:
						image = image["url"]
						s += f"<{image}>\n"
					thumbnail = e.get("thumbnail")
					if thumbnail:
						thumbnail = thumbnail["url"]
						s += f"<{thumbnail}>\n"
					if e.get("fields"):
						s += "\t".join(f["name"] for f in e["fields"]) + "\t".join(f["value"] for f in e["fields"])
					s += "}"
				s += "\n\u200c\n"
				found = ereg.findall(s)
				for er in found:
					try:
						id = int(er.removesuffix(">").rsplit(":", 1)[-1])
					except:
						print_exc()
					else:
						animated = er.startswith("<a:")
						name = er.split(":", 2)[1]
						if id not in emojis or len(emojis[id]["name"]) < len(name):
							emojis[id] = dict(id=id, name=name, animated=animated)
				f.write(s.encode("utf-8"))
			time.sleep(1)
			if len(d) < 100:
				break
	with open(sfold + cfold + cname + ".json", "w") as f:
		json.dump(data, f)
	# break

if users and not os.path.exists(sfold + "users"):
	os.mkdir(sfold + "users")
for uid, u in users.items():
	name = u["username"] + " ~ " + uid
	avatar = u["avatar"]
	url = f"https://cdn.discordapp.com/avatars/{uid}/{avatar}.png?size=4096"
	futs.append(exc.submit(download, url, f"{sfold}users/{name}"))

for fut in futs:
	fut.result()
futs.clear()

if emojis:
	if not os.path.exists(sfold + "emojis"):
		os.mkdir(sfold + "emojis")
	for emoji in emojis.values():
		eid = emoji["id"]
		name = emoji["name"] + " ~ " + str(eid)
		fmt = "gif" if emoji.get("animated", True) else "png"
		url = f"https://cdn.discordapp.com/emojis/{eid}.{fmt}"
		futs.append(exc.submit(download, url, f"{sfold}emojis/{name}", backup=url.rsplit(".", 1)[0] + ".png" if emoji.get("animated", True) else None))

for fut in futs:
	fut.result()

if fn:
	import zipfile, shutil
	from pathlib import Path
	if os.path.exists(fn):
		os.remove(fn)
	fold = Path(sfold)
	with zipfile.ZipFile(fn, "w", zipfile.ZIP_DEFLATED) as z:
		for entry in fold.rglob("*"):
			z.write(entry, entry.relative_to(fold), compress_type=zipfile.ZIP_LZMA if str(entry).rsplit(".", 1)[-1] in compressible else zipfile.ZIP_STORED, compresslevel=6)
	shutil.rmtree(sfold)

sys.stdout.write(f"\rProgress: {PROGRESS} (Complete){END}")