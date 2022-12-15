import sys, os

if len(sys.argv) < 3:
	print(f"Usage: python {sys.argv[0]} <token> <server-id>")
	sys.exit(1)

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

import requests, json, time, concurrent.futures, random
reqs = requests.Session()
token, gid = sys.argv[1], sys.argv[2]

exc = concurrent.futures.ThreadPoolExecutor(max_workers=64)
futs = []
PTIME = 0

headers = {"Authorization": "Bot " + token, "Content-Type": "application/json"}
resp = reqs.get(
	"https://discord.com/api/v9/users/@me/guilds",
	headers=headers,
)
PROGRESS += 1
if resp.status_code == 401:
	headers = {"Authorization": token, "Content-Type": "application/json"}
	resp = reqs.get(
		"https://discord.com/api/v9/users/@me/guilds",
		headers=headers,
	)
	PROGRESS += 1
resp.raise_for_status()
d = resp.json()

server = [s for s in d if int(gid) == int(s["id"])]
if not server:
	raise FileNotFoundError("Server not found.")
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

resp = reqs.get(
	f"https://discord.com/api/v9/guilds/{gid}/channels",
	headers=headers,
)
PROGRESS += 1
channels = resp.json()

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
				from traceback import print_exc
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
						from traceback import print_exc
						print_exc()
			pos = os.path.getsize(fn)
			if pos < size:
				raise EOFError(f"{url}: Incomplete download ({pos} < {size}), unable to resolve.")
		return fn

def download(url, fn):
	try:
		resp = proxy_download(url)
	except:
		from traceback import print_exc
		print_exc()
		resp = reqs.get(url)
	globals()["PROGRESS"] += 1
	if resp.headers.get("Content-Type"):
		ctype = resp.headers["Content-Type"]
		fn += "." + map_mime(ctype)
	with open(fn, "wb") as f:
		f.write(resp.content)
	t = time.time()
	if t - PTIME >= 1:
		globals()["PTIME"] = t
		print(f"\rProgress: {PROGRESS}", end=END)

users = {}
while channels:
	channel = channels.pop(0)
	if channel["type"] not in (0, 1, 3, 5, 10, 11, 12, 15):
		continue
	cid = channel["id"]
	for mode in ("public", "private"):
		resp = reqs.get(
			f"https://discord.com/api/v9/channels/{cid}/threads/archived/{mode}",
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
				f"https://discord.com/api/v9/channels/{cid}/messages?limit=100&after={MID}",
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

resp = reqs.get(
	f"https://discord.com/api/v9/guilds/{gid}/emojis",
	headers=headers,
)
emojis = resp.json()

if emojis:
	if not os.path.exists(sfold + "emojis"):
		os.mkdir(sfold + "emojis")
	for emoji in emojis:
		eid = emoji["id"]
		name = emoji["name"] + " ~ " + eid
		fmt = "gif" if emoji.get("animated") else "png"
		url = f"https://cdn.discordapp.com/emojis/{eid}.{fmt}?quality=lossless"
		futs.append(exc.submit(download, url, f"{sfold}emojis/{name}"))

for fut in futs:
	fut.result()

if fn:
	import zipfile, shutil
	from pathlib import Path
	if os.path.exists(fn):
		os.remove(fn)
	fold = Path(sfold + os.listdir(sfold)[0])
	with zipfile.ZipFile(fn, "w", zipfile.ZIP_STORED) as z:
		for entry in fold.rglob("*"):
			z.write(entry, entry.relative_to(fold))
	shutil.rmtree(sfold)

print(f"\rProgress: {PROGRESS} (Complete)", end=END)