# This file mostly contains code copied from the Miza discord bot's voice command category

import concurrent.futures
from concurrent.futures import thread

def _adjust_thread_count(self):
	# if idle threads are available, don't spin new threads
	try:
		if self._idle_semaphore.acquire(timeout=0):
			return
	except AttributeError:
		pass

	# When the executor gets lost, the weakref callback will wake up
	# the worker threads.
	def weakref_cb(_, q=self._work_queue):
		q.put(None)

	num_threads = len(self._threads)
	if num_threads < self._max_workers:
		thread_name = '%s_%d' % (self._thread_name_prefix or self, num_threads)
		t = thread.threading.Thread(
			name=thread_name,
			target=thread._worker,
			args=(
				thread.weakref.ref(self, weakref_cb),
				self._work_queue,
				self._initializer,
				self._initargs,
			),
			daemon=True
		)
		t.start()
		self._threads.add(t)
		thread._threads_queues[t] = self._work_queue

concurrent.futures.ThreadPoolExecutor._adjust_thread_count = lambda self: _adjust_thread_count(self)

exc = concurrent.futures.ThreadPoolExecutor(max_workers=32)
create_future_ex = exc.submit

class MultiAutoImporter:

	class ImportedModule:

		def __init__(self, module, pool, _globals):
			object.__setattr__(self, "__module", module)
			object.__setattr__(self, "__fut", pool.submit(__import__, module))
			object.__setattr__(self, "__globals", _globals)

		def __getattr__(self, k):
			m = self.force()
			return getattr(m, k)

		def __setattr__(self, k, v):
			m = self.force()
			return setattr(m, k, v)

		def force(self):
			module = object.__getattribute__(self, "__module")
			_globals = object.__getattribute__(self, "__globals")
			_globals[module] = m = object.__getattribute__(self, "__fut").result()
			return m

	def __init__(self, *args, pool=None, _globals=None):
		self.pool = pool
		if not _globals:
			_globals = globals()
		args = " ".join(args).replace(",", " ").split()
		if not pool:
			_globals.update((k, __import__(k)) for k in args)
		else:
			futs = []
			for arg in args:
				futs.append(self.ImportedModule(arg, pool, _globals))
			_globals.update(zip(args, futs))

importer = MultiAutoImporter(
	"contextlib, urllib, collections, math, traceback, requests, orjson",
	"os, yt_dlp, random, time, base64, hashlib, re, psutil, subprocess, itertools, zipfile",
	pool=exc,
	_globals=globals(),
)

suppress = contextlib.suppress
urllib.force()
import urllib.parse
url_parse = urllib.parse.quote_plus
deque = collections.deque
math.force()
from math import *
traceback.force()
from traceback import print_exc
reqs = requests.Session()
youtube_dl = yt_dlp.force()


class cdict(dict):

	__slots__ = ()

	__init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
	__repr__ = lambda self: f"{self.__class__.__name__}({super().__repr__() if super().__len__() else ''})"
	__str__ = lambda self: super().__repr__()
	__iter__ = lambda self: iter(tuple(super().__iter__()))
	__call__ = lambda self, k: self.__getitem__(k)

	def __getattr__(self, k):
		with suppress(AttributeError):
			return self.__getattribute__(k)
		if not k.startswith("__") or not k.endswith("__"):
			try:
				return self.__getitem__(k)
			except KeyError as ex:
				raise AttributeError(*ex.args)
		raise AttributeError(k)

	def __setattr__(self, k, v):
		if k.startswith("__") and k.endswith("__"):
			return object.__setattr__(self, k, v)
		return self.__setitem__(k, v)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(self)
		return data

	def union(self, other=None, **kwargs):
		temp = self.copy()
		if other:
			temp.update(other)
		if kwargs:
			temp.update(kwargs)
		return temp

	@property
	def __dict__(self):
		return self

	___repr__ = lambda self: super().__repr__()
	copy = __copy__ = lambda self: self.__class__(self)
	to_dict = lambda self: dict(**self)
	to_list = lambda self: list(super().values())


class demap(collections.abc.Mapping):

	__slots__ = ("a", "b")

	def __init__(self, *args, **kwargs):
		self.a = cdict(*args, **kwargs)
		self.b = cdict(reversed(t) for t in self.a.items())

	def __getitem__(self, k):
		with suppress(KeyError):
			return self.a.__getitem__(k)
		return self.b.__getitem__(k)

	def __delitem__(self, k):
		try:
			temp = self.a.pop(k)
		except KeyError:
			temp = self.b.pop(k)
			if temp in self.a:
				self.__delitem__(temp)
		else:
			if temp in self.b:
				self.__delitem__(temp)
		return self

	def __setitem__(self, k, v):
		if k not in self.a:
			if v not in self.a:
				self.a.__setitem__(k, v)
				self.b.__setitem__(v, k)
			else:
				self.__delitem__(v)
				self.__setitem__(k, v)
		else:
			self.__delitem__(k)
			if v in self.a:
				self.__delitem__(v)
			self.__setitem__(k, v)
		return self

	def get(self, k, v=None):
		with suppress(KeyError):
			return self.__getitem__(k)
		return v
	
	def pop(self, k, v=None):
		with suppress(KeyError):
			temp = self.__getitem__(k)
			self.__delitem__(k)
			return temp
		return v

	def popitem(self, k, v=None):
		with suppress(KeyError):
			temp = self.__getitem__(k)
			self.__delitem__(k)
			return (k, temp)
		return v

	clear = lambda self: (self.a.clear(), self.b.clear())
	__bool__ = lambda self: bool(self.a)
	__iter__ = lambda self: iter(self.a.items())
	__reversed__ = lambda self: reversed(self.a.items())
	__len__ = lambda self: self.b.__len__()
	__str__ = lambda self: self.a.__str__()
	__repr__ = lambda self: f"{self.__class__.__name__}({self.a.__repr__() if bool(self.b) else ''})"
	__contains__ = lambda self, k: k in self.a or k in self.b


class delay(contextlib.AbstractContextManager, contextlib.ContextDecorator, collections.abc.Callable):

	def __init__(self, duration=0):
		self.duration = duration
		self.start = utc()

	def __call__(self):
		return self.exit()
	
	def __exit__(self, *args):
		remaining = self.duration - utc() + self.start
		if remaining > 0:
			time.sleep(remaining)


def exclusive_range(range, *excluded):
	ex = frozenset(excluded)
	return tuple(i for i in range if i not in ex)


ZeroEnc = "\xad\u061c\u180e\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\u2065\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f\ufe0f\ufeff"
__zeroEncoder = demap({chr(i + 97): c for i, c in enumerate(ZeroEnc)})
__zeroEncode = "".maketrans(dict(__zeroEncoder.a))
__zeroDecode = "".maketrans(dict(__zeroEncoder.b))
is_zero_enc = lambda s: (s[0] in ZeroEnc) if s else None
zwencode = lambda s: (s if type(s) is str else str(s)).casefold().translate(__zeroEncode)
zwdecode = lambda s: (s if type(s) is str else str(s)).casefold().translate(__zeroDecode)


# Unicode fonts for alphanumeric characters.
UNIFMTS = [
	"𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙",
	"𝟢𝟣𝟤𝟥𝟦𝟧𝟨𝟩𝟪𝟫𝓪𝓫𝓬𝓭𝓮𝓯𝓰𝓱𝓲𝓳𝓴𝓵𝓶𝓷𝓸𝓹𝓺𝓻𝓼𝓽𝓾𝓿𝔀𝔁𝔂𝔃𝓐𝓑𝓒𝓓𝓔𝓕𝓖𝓗𝓘𝓙𝓚𝓛𝓜𝓝𝓞𝓟𝓠𝓡𝓢𝓣𝓤𝓥𝓦𝓧𝓨𝓩",
	"𝟢𝟣𝟤𝟥𝟦𝟧𝟨𝟩𝟪𝟫𝒶𝒷𝒸𝒹𝑒𝒻𝑔𝒽𝒾𝒿𝓀𝓁𝓂𝓃𝑜𝓅𝓆𝓇𝓈𝓉𝓊𝓋𝓌𝓍𝓎𝓏𝒜𝐵𝒞𝒟𝐸𝐹𝒢𝐻𝐼𝒥𝒦𝐿𝑀𝒩𝒪𝒫𝒬𝑅𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵",
	"𝟘𝟙𝟚𝟛𝟜𝟝𝟞𝟟𝟠𝟡𝕒𝕓𝕔𝕕𝕖𝕗𝕘𝕙𝕚𝕛𝕜𝕝𝕞𝕟𝕠𝕡𝕢𝕣𝕤𝕥𝕦𝕧𝕨𝕩𝕪𝕫𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄ℕ𝕆ℙℚℝ𝕊𝕋𝕌𝕍𝕎𝕏𝕐ℤ",
	"0123456789𝔞𝔟𝔠𝔡𝔢𝔣𝔤𝔥𝔦𝔧𝔨𝔩𝔪𝔫𝔬𝔭𝔮𝔯𝔰𝔱𝔲𝔳𝔴𝔵𝔶𝔷𝔄𝔅ℭ𝔇𝔈𝔉𝔊ℌℑ𝔍𝔎𝔏𝔐𝔑𝔒𝔓𝔔ℜ𝔖𝔗𝔘𝔙𝔚𝔛𝔜ℨ",
	"0123456789𝖆𝖇𝖈𝖉𝖊𝖋𝖌𝖍𝖎𝖏𝖐𝖑𝖒𝖓𝖔𝖕𝖖𝖗𝖘𝖙𝖚𝖛𝖜𝖝𝖞𝖟𝕬𝕭𝕮𝕯𝕰𝕱𝕲𝕳𝕴𝕵𝕶𝕷𝕸𝕹𝕺𝕻𝕼𝕽𝕾𝕿𝖀𝖁𝖂𝖃𝖄𝖅",
	"０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
	#"0123456789ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘQʀꜱᴛᴜᴠᴡxʏᴢᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘQʀꜱᴛᴜᴠᴡxʏᴢ",
	"⓪①②③④⑤⑥⑦⑧⑨🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉",
	"⓿➊➋➌➍➎➏➐➑➒🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉",
	"⓪①②③④⑤⑥⑦⑧⑨ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ",
	"⓿➊➋➌➍➎➏➐➑➒🅐🅑🅒🅓🅔🅕🅖🅗🅘🅙🅚🅛🅜🅝🅞🅟🅠🅡🅢🅣🅤🅥🅦🅧🅨🅩🅐🅑🅒🅓🅔🅕🅖🅗🅘🅙🅚🅛🅜🅝🅞🅟🅠🅡🅢🅣🅤🅥🅦🅧🅨🅩",
	"0123456789𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡",
	"𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕",
	"𝟶𝟷𝟸𝟹𝟺𝟻𝟼𝟽𝟾𝟿𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉",
	"₀₁₂₃₄₅₆₇₈₉ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖqʳˢᵗᵘᵛʷˣʸᶻ🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿",
	"0123456789ᗩᗷᑢᕲᘿᖴᘜᕼᓰᒚҠᒪᘻᘉᓍᕵᕴᖇSᖶᑘᐺᘺ᙭ᖻᗱᗩᗷᑕᗪᗴᖴǤᕼIᒍKᒪᗰᑎOᑭᑫᖇᔕTᑌᐯᗯ᙭Yᘔ",
	"0ƖᘔƐᔭ59Ɫ86ɐqɔpǝɟɓɥᴉſʞןɯuodbɹsʇnʌʍxʎzꓯᗺƆᗡƎℲ⅁HIſꓘ⅂WNOԀΌᴚS⊥∩ΛMX⅄Z",
	"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
]
__umap = {UNIFMTS[k][i]: UNIFMTS[-1][i] for k in range(len(UNIFMTS) - 1) for i in range(len(UNIFMTS[k]))}

__unfont = "".maketrans(__umap)
unfont = lambda s: str(s).translate(__unfont)

DIACRITICS = {
	"ÀÁÂÃÄÅĀĂĄ": "A",
	"Æ": "AE",
	"ÇĆĈĊČ": "C",
	"ĎĐ": "D",
	"ÈÉÊËĒĔĖĘĚ": "E",
	"ĜĞĠĢ": "G",
	"ĤĦ": "H",
	"ÌÍÎÏĨĪĬĮİ": "I",
	"Ĳ": "IJ",
	"Ĵ": "J",
	"Ķ": "K",
	"ĹĻĽĿŁ": "L",
	"ÑŃŅŇŊ": "N",
	"ÒÓÔÕÖØŌŎŐ": "O",
	"Œ": "OE",
	"ŔŖŘ": "R",
	"ŚŜŞŠ": "S",
	"ŢŤŦ": "T",
	"ÙÚÛÜŨŪŬŮŰŲ": "U",
	"Ŵ": "W",
	"ÝŶŸ": "Y",
	"ŹŻŽ": "Z",
	"àáâãäåāăą": "a",
	"æ": "ae",
	"çćĉċč": "c",
	"ďđ": "d",
	"èéêëðēĕėęě": "e",
	"ĝğġģ": "g",
	"ĥħ": "h",
	"ìíîïĩīĭįı": "i",
	"ĳ": "ij",
	"ĵ": "j",
	"ķĸ": "k",
	"ĺļľŀł": "l",
	"ñńņňŉŋ": "n",
	"òóôõöøōŏő": "o",
	"œ": "oe",
	"þ": "p",
	"ŕŗř": "r",
	"śŝşšſ": "s",
	"ß": "ss",
	"ţťŧ": "t",
	"ùúûüũūŭůűų": "u",
	"ŵ": "w",
	"ýÿŷ": "y",
	"źżž": "z",
}
for i, k in DIACRITICS.items():
	__umap.update({c: k for c in i})
__umap.update({c: "" for c in ZeroEnc})
__umap["\u200a"] = ""
for c in tuple(__umap):
	if c in UNIFMTS[-1]:
		__umap.pop(c)
__trans = "".maketrans(__umap)
extra_zalgos = (
	range(768, 880),
	range(1155, 1162),
	exclusive_range(range(1425, 1478), 1470, 1472, 1475),
	range(1552, 1560),
	range(1619, 1632),
	exclusive_range(range(1750, 1774), 1757, 1758, 1765, 1766, 1769),
	exclusive_range(range(2260, 2304), 2274),
	range(7616, 7627),
	(8432,),
	range(11744, 11776),
	(42607,), range(42612, 42622), (42654, 42655),
	range(65056, 65060),
)
zalgo_array = list(itertools.chain(*extra_zalgos))
zalgo_map = {n: "" for n in zalgo_array}
__trans.update(zalgo_map)
__unitrans = ["".maketrans({UNIFMTS[-1][x]: UNIFMTS[i][x] for x in range(len(UNIFMTS[-1]))}) for i in range(len(UNIFMTS) - 1)]

# Translates all alphanumeric characters in a string to their corresponding character in the desired font.
def uni_str(s, fmt=0):
	if type(s) is not str:
		s = str(s)
	return s.translate(__unitrans[fmt])

# Translates all alphanumeric characters in unicode fonts to their respective ascii counterparts.
def unicode_prune(s):
	if type(s) is not str:
		s = str(s)
	if s.isascii():
		return s
	return s.translate(__trans)

__qmap = {
	"“": '"',
	"”": '"',
	"„": '"',
	"‘": "'",
	"’": "'",
	"‚": "'",
	"〝": '"',
	"〞": '"',
	"⸌": "'",
	"⸍": "'",
	"⸢": "'",
	"⸣": "'",
	"⸤": "'",
	"⸥": "'",
}
__qtrans = "".maketrans(__qmap)

full_prune = lambda s: unicode_prune(s).translate(__qtrans).casefold()


# A fuzzy substring search that returns the ratio of characters matched between two strings.
def fuzzy_substring(sub, s, match_start=False, match_length=True):
	if not match_length and s in sub:
		return 1
	match = 0
	if not match_start or sub and s.startswith(sub[0]):
		found = [0] * len(s)
		x = 0
		for i, c in enumerate(sub):
			temp = s[x:]
			if temp.startswith(c):
				if found[x] < 1:
					match += 1
					found[x] = 1
				x += 1
			elif c in temp:
				y = temp.index(c)
				x += y
				if found[x] < 1:
					found[x] = 1
					match += 1 - y / len(s)
				x += 1
			else:
				temp = s[:x]
				if c in temp:
					y = temp.rindex(c)
					if found[y] < 1:
						match += 1 - (x - y) / len(s)
						found[y] = 1
					x = y + 1
		if len(sub) > len(s) and match_length:
			match *= len(s) / len(sub)
	# ratio = match / len(s)
	ratio = max(0, min(1, match / len(s)))
	return ratio


# Replaces words in a string from a mapping similar to str.replace, but performs operation both ways.
def replace_map(s, mapping):
	temps = {k: chr(65535 - i) for i, k in enumerate(mapping.keys())}
	trans = "".maketrans({chr(65535 - i): mapping[k] for i, k in enumerate(mapping.keys())})
	for key, value in temps.items():
		s = s.replace(key, value)
	for key, value in mapping.items():
		s = s.replace(value, key)
	return s.translate(trans)


def belongs(s):
	s = s.strip()
	if not s:
		return ""
	if full_prune(s[-1]) == "s":
		return s + "'"
	return s + "'s"


def parse_fs(fs):
	if type(fs) is not bytes:
		fs = str(fs).encode("utf-8")
	if fs.endswith(b"TB"):
		scale = 1099511627776
	if fs.endswith(b"GB"):
		scale = 1073741824
	elif fs.endswith(b"MB"):
		scale = 1048576
	elif fs.endswith(b"KB"):
		scale = 1024
	else:
		scale = 1
	return float(fs.split(None, 1)[0]) * scale

RE = cdict()

def regexp(s, flags=0):
	global RE
	if issubclass(type(s), re.Pattern):
		return s
	elif type(s) is not str:
		s = s.decode("utf-8", "replace")
	t = (s, flags)
	try:
		return RE[t]
	except KeyError:
		RE[t] = re.compile(s, flags)
		return RE[t]

def strip_acc(url):
	if url.startswith("<") and url[-1] == ">":
		s = url[1:-1]
		if is_url(s):
			return s
	return url

def html_decode(s):
	while len(s) > 7:
		try:
			i = s.index("&#")
		except ValueError:
			break
		try:
			if s[i + 2] == "x":
				h = "0x"
				p = i + 3
			else:
				h = ""
				p = i + 2
			for a in range(4):
				if s[p + a] == ";":
					v = int(h + s[p:p + a])
					break
			c = chr(v)
			s = s[:i] + c + s[p + a + 1:]
		except ValueError:
			continue
		except IndexError:
			continue
	s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
	return s.replace("&quot;", '"').replace("&apos;", "'")

def time_parse(ts):
	data = str(ts).split(":")
	t = 0
	mult = 1
	while len(data):
		t += float(data[-1]) * mult
		data = data[:-1]
		if mult <= 60:
			mult *= 60
		elif mult <= 3600:
			mult *= 24
		elif len(data):
			raise TypeError("Too many time arguments.")
	return t


utc = lambda: time.time_ns() / 1e9

single_space = lambda s: regexp("  +").sub(" ", s)
__smap = {"|": "", "*": ""}
__strans = "".maketrans(__smap)
verify_search = lambda f: strip_acc(single_space(f.strip().translate(__strans)))
find_urls = lambda url: regexp("(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+").findall(url)
is_url = lambda url: regexp("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+$").fullmatch(url)
is_discord_url = lambda url: regexp("^https?:\\/\\/(?:[A-Za-z]{3,8}\\.)?discord(?:app)?\\.(?:com|net)\\/").findall(url) + regexp("https:\\/\\/images-ext-[0-9]+\\.discordapp\\.net\\/external\\/").findall(url)
is_tenor_url = lambda url: regexp("^https?:\\/\\/tenor.com(?:\\/view)?/[a-zA-Z0-9\\-_]+-[0-9]+").findall(url)
is_imgur_url = lambda url: regexp("^https?:\\/\\/(?:[A-Za-z]\\.)?imgur.com/[a-zA-Z0-9\\-_]+").findall(url)
is_giphy_url = lambda url: regexp("^https?:\\/\\/giphy.com/gifs/[a-zA-Z0-9\\-_]+").findall(url)
is_youtube_url = lambda url: regexp("^https?:\\/\\/(?:www\\.)?youtu(?:\\.be|be\\.com)\\/[^\\s<>`|\"']+").findall(url)
is_youtube_stream = lambda url: url and re.findall(r"^https?:\/\/r+[0-9]+---.{2}-[A-Za-z0-9\-_]{4,}\.googlevideo\.com", url)
is_deviantart_url = lambda url: regexp("^https?:\\/\\/(?:www\\.)?deviantart\\.com\\/[^\\s<>`|\"']+").findall(url)

def expired(stream):
	if is_youtube_url(stream):
		return True
	if stream.startswith("https://www.yt-download.org/download/"):
		if int(stream.split("/download/", 1)[1].split("/", 4)[3]) < utc() + 60:
			return True
	elif is_youtube_stream(stream):
		if int(stream.replace("/", "=").split("expire=", 1)[-1].split("=", 1)[0].split("&", 1)[0]) < utc() + 60:
			return True

verify_url = lambda url: url if is_url(url) else url_parse(url)

is_alphanumeric = lambda string: string.replace(" ", "").isalnum()
to_alphanumeric = lambda string: single_space(regexp("[^a-z 0-9]", re.I).sub(" ", unicode_prune(string)))

shash = lambda s: base64.b64encode(hashlib.sha256(s.encode("utf-8")).digest()).replace(b"/", b"-").decode("utf-8", "replace")


def round_min(x):
	if type(x) is str:
		if "." in x:
			x = x.strip("0")
			if len(x) > 8 and "mpf" in globals():
				x = mpf(x)
			else:
				x = float(x)
		else:
			try:
				return int(x)
			except ValueError:
				return float(x)
	if type(x) is int:
		return x
	if type(x) is not complex:
		if isfinite(x):
			if type(x) is globals().get("mpf", None):
				y = int(x)
				if x == y:
					return y
				f = float(x)
				if str(x) == str(f):
					return f
			else:
				y = math.round(x)
				if x == y:
					return int(y)
		return x
	else:
		if x.imag == 0:
			return round_min(x.real)
		else:
			return round_min(complex(x).real) + round_min(complex(x).imag) * (1j)
math.round = round


eval_json = lambda s: eval(s, dict(true=True, false=False, null=None, none=None), {})


IMAGE_FORMS = {
	".gif": True,
	".png": True,
	".bmp": False,
	".jpg": True,
	".jpeg": True,
	".tiff": False,
	".webp": True,
}
def is_image(url):
	url = url.split("?", 1)[0]
	if "." in url:
		url = url[url.rindex("."):]
		url = url.casefold()
		return IMAGE_FORMS.get(url)


# Gets estimated duration from duration stored in queue entry
e_dur = lambda d: float(d) if type(d) is str else (d if d is not None else 300)


ffprobe = "ffprobe.exe" if os.name == "nt" else "ffprobe"

# Runs ffprobe on a file or url, returning the duration if possible.
def _get_duration(filename, _timeout=12):
	command = (
		ffprobe,
		"-v",
		"error",
		"-select_streams",
		"a:0",
		"-show_entries",
		"format=duration,bit_rate",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
		filename,
	)
	resp = None
	try:
		proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
		fut = create_future_ex(proc.wait, timeout=_timeout)
		res = fut.result(timeout=_timeout)
		resp = proc.stdout.read().split()
	except:
		with suppress():
			proc.kill()
		print_exc()
	try:
		dur = float(resp[0])
	except (IndexError, ValueError, TypeError):
		dur = None
	bps = None
	if resp and len(resp) > 1:
		with suppress(ValueError):
			bps = float(resp[1])
	return dur, bps

def get_duration(filename):
	if filename:
		dur, bps = _get_duration(filename, 4)
		if not dur and is_url(filename):
			with reqs.head(filename) as resp:
				head = {k.casefold(): v for k, v in resp.headers.items()}
				if "content-length" not in head:
					return _get_duration(filename, 20)[0]
				if bps:
					print(head, bps, sep="\n")
					return (int(head["content-length"]) << 3) / bps
				ctype = [e.strip() for e in head.get("content-type", "").split(";") if "/" in e][0]
				if ctype.split("/", 1)[0] not in ("audio", "video"):
					return nan
				if ctype == "audio/midi":
					return nan
		return dur


# Gets the best icon/thumbnail for a queue entry.
def get_best_icon(entry):
	with suppress(KeyError):
		return entry["thumbnail"]
	with suppress(KeyError):
		return entry["icon"]
	try:
		thumbnails = entry["thumbnails"]
	except KeyError:
		try:
			url = entry["webpage_url"]
		except KeyError:
			url = entry["url"]
		if not url:
			return ""
		if is_discord_url(url):
			if not is_image(url):
				return "https://cdn.discordapp.com/embed/avatars/0.png"
		if is_youtube_url(url):
			if "?v=" in url:
				vid = url.split("?v=", 1)[-1]
			else:
				vid = url.rsplit("/", 1)[-1].split("?", 1)[0]
			entry["thumbnail"] = f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg"
			return entry["thumbnail"]
		return url
	return sorted(thumbnails, key=lambda x: float(x.get("width", x.get("preference", 0) * 4096)), reverse=True)[0]["url"]

# Gets the best audio file download link for a queue entry.
def get_best_audio(entry):
	with suppress(KeyError):
		return entry["stream"]
	best = -inf
	try:
		fmts = entry["formats"]
	except KeyError:
		fmts = ()
	try:
		url = entry["webpage_url"]
	except KeyError:
		url = entry["url"]
	replace = True
	for fmt in fmts:
		q = fmt.get("abr", 0)
		if not isinstance(q, (int, float)):
			q = 0
		if q <= 0:
			if fmt.get("asr"):
				q = fmt["asr"] / 1000
			elif fmt.get("audio_channels"):
				q = fmt["audio_channels"]
		vcodec = fmt.get("vcodec", "none")
		if vcodec not in (None, "none"):
			q -= 1
		if not fmt["url"].startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
			replace = False
		# print(fmt["url"], q)
		if q > best or replace:
			best = q
			url = fmt["url"]
	if "dropbox.com" in url:
		if "?dl=0" in url:
			url = url.replace("?dl=0", "?dl=1")
	if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
		resp = reqs.get(url).content
		fmts = deque()
		with suppress(ValueError, KeyError):
			while True:
				search = b'<Representation id="'
				resp = resp[resp.index(search) + len(search):]
				f_id = resp[:resp.index(b'"')].decode("utf-8")
				search = b"><BaseURL>"
				resp = resp[resp.index(search) + len(search):]
				stream = resp[:resp.index(b'</BaseURL>')].decode("utf-8")
				fmt = cdict(youtube_dl.extractor.youtube.YoutubeIE._formats[f_id])
				fmt.url = stream
				fmts.append(fmt)
		entry["formats"] = fmts
		return get_best_audio(entry)
	if not url:
		raise KeyError("URL not found.")
	return url

# Gets the best video file download link for a queue entry.
def get_best_video(entry):
	with suppress(KeyError):
		return entry["stream"]
	best = -inf
	try:
		fmts = entry["formats"]
	except KeyError:
		fmts = ()
	try:
		url = entry["webpage_url"]
	except KeyError:
		url = entry.get("url")
	replace = True
	for fmt in fmts:
		q = 720 - abs(720 - (fmt.get("height") or 0))
		if not isinstance(q, (int, float)):
			q = 0
		vcodec = fmt.get("vcodec", "none")
		if vcodec in (None, "none"):
			q -= inf
		u = fmt["url"]
		if not u.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
			replace = False
		if q > best or replace:
			best = q
			url = fmt["url"]
	if "dropbox.com" in url:
		if "?dl=0" in url:
			url = url.replace("?dl=0", "?dl=1")
	if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
		resp = reqs.get(url).content
		fmts = deque()
		with suppress(ValueError, KeyError):
			while True:
				search = b'<Representation id="'
				resp = resp[resp.index(search) + len(search):]
				f_id = resp[:resp.index(b'"')].decode("utf-8")
				search = b"><BaseURL>"
				resp = resp[resp.index(search) + len(search):]
				stream = resp[:resp.index(b'</BaseURL>')].decode("utf-8")
				fmt = cdict(youtube_dl.extractor.youtube.YoutubeIE._formats[f_id])
				fmt.url = stream
				fmts.append(fmt)
		entry["formats"] = fmts
		return get_best_video(entry)
	if not url:
		raise KeyError("URL not found.")
	return url


# Manages all audio searching and downloading.
class AudioDownloader:

	_globals = globals()
	ydl_opts = {
		# "verbose": 1,
		"quiet": 1,
		"format": "bestaudio/best",
		"nocheckcertificate": 1,
		"no_call_home": 1,
		"nooverwrites": 1,
		"noplaylist": 1,
		"logtostderr": 0,
		"ignoreerrors": 0,
		"default_search": "auto",
		"source_address": "0.0.0.0",
	}
	youtube_x = 0
	youtube_dl_x = 0
	spotify_x = 0
	other_x = 0

	def __init__(self):
		self.bot = None
		self.lastclear = 0
		self.downloading = cdict()
		self.cache = cdict()
		self.searched = cdict()
		self.waiting = create_future_ex(self.update_dl)
		self.youtube_base = "CONSENT=YES+cb.20210328-17-p0.en+FX"
		self.cookie = create_future_ex(self.set_cookie)
		self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)

	def set_cookie(self):
		resp = reqs.get("https://www.youtube.com").text
		if "<title>Before you continue to YouTube</title>" in resp:
			resp = resp.split('<input type="hidden" name="v" value="', 1)[-1]
			resp = resp[:resp.index('">')].rsplit("+", 1)[0]
			self.youtube_base = f"CONSENT=YES+{resp}"
			self.youtube_x += 1
	
	def youtube_header(self):
		return {
			"Cookie": self.youtube_base + "%03d" % random.randint(0, 999) + ";",
			"User-Agent": f"Mozilla/5.{random.randint(1, 9)}",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
		}

	# Initializes youtube_dl object as well as spotify tokens, every 720 seconds.
	def update_dl(self):
		try:
			self.youtube_dl_x += 1
			self.spotify_x += 1
			token = reqs.get("https://open.spotify.com/get_access_token").content
			self.spotify_header = {"authorization": f"Bearer {orjson.loads(token[:512])['accessToken']}"}
			self.other_x += 1
		except:
			pass

	# Gets data from yt-download.org, and adjusts the format to ensure compatibility with results from youtube-dl. Used as backup.
	def extract_backup(self, url):
		url = verify_url(url)
		if is_url(url) and not is_youtube_url(url):
			with reqs.head(url) as resp:
				url = resp.url
				name = url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
				ctype = resp.headers.get("Content-Type")
				if ctype.startswith("video") or ctype.startswith("audio"):
					return dict(
						id=name,
						title=name,
						direct=True,
						url=url,
						webpage_url=url,
						extractor="generic",
					)
				elif ctype == "application/octet-stream":
					dur, bps = get_duration(url)
					d = dict(
						id=name,
						title=name,
						direct=True,
						url=url,
						webpage_url=url,
						extractor="generic",
					)
					if dur:
						d["duration"] = dur
					return d
			raise TypeError("Not a youtube link.")
		excs = deque()
		if ":" in url:
			url = url.rsplit("/", 1)[-1].split("v=", 1)[-1].split("&", 1)[0]
		webpage_url = f"https://www.youtube.com/watch?v={url}"
		resp = None
		try:
			yt_url = f"https://www.yt-download.org/file/mp3/{url}"
			self.other_x += 1
			resp = reqs.get(yt_url, timeout=60).content
			search = b'<img class="h-20 w-20 md:h-48 md:w-48 mt-0 md:mt-12 lg:mt-0 rounded-full mx-auto md:mx-0 md:mr-6" src="'
			resp = resp[resp.index(search) + len(search):]
			thumbnail = resp[:resp.index(b'"')].decode("utf-8", "replace")
			search = b'<h2 class="text-lg text-teal-600 font-bold m-2 text-center">'
			resp = resp[resp.index(search) + len(search):]
			title = html_decode(resp[:resp.index(b"</h2>")].decode("utf-8", "replace"))
			resp = resp[resp.index(f'<a href="https://www.yt-download.org/download/{url}/mp3/256'.encode("utf-8")) + 9:]
			stream = resp[:resp.index(b'"')].decode("utf-8", "replace")
			resp = resp[:resp.index(b"</a>")]
			search = b'<div class="text-shadow-1">'
			fs = parse_fs(resp[resp.rindex(search) + len(search):resp.rindex(b"</div>")])
			dur = fs / 192000 * 8
			entry = {
				"formats": [
					{
						"abr": 256,
						"url": stream,
					},
				],
				"duration": dur,
				"thumbnail": thumbnail,
				"title": title,
				"webpage_url": webpage_url,
			}
			print("Successfully resolved with yt-download.")
			return entry
		except Exception as ex:
			if resp:
				try:
					search = b'<h3 class="text-center text-xl">'
					resp = resp[resp.index(search) + len(search):]
					resp = resp[:resp.index(b"<")]
				except ValueError:
					pass
				else:
					excs.append(resp.decode("utf-8"))
			excs.append(ex)
			print(excs)
			raise

	# Returns part of a spotify playlist.
	def get_spotify_part(self, url):
		out = deque()
		self.spotify_x += 1
		self.waiting.result()
		resp = reqs.get(url, headers=self.spotify_header).content
		d = orjson.loads(resp)
		with suppress(KeyError):
			d = d["tracks"]
		try:
			items = d["items"]
			total = d.get("total", 0)
		except KeyError:
			if "type" in d:
				items = (d,)
				total = 1
			else:
				items = []
				total = 0
		for item in items:
			try:
				track = item["track"]
			except KeyError:
				try:
					track = item["episode"]
				except KeyError:
					if "id" in item:
						track = item
					else:
						continue
			name = track.get("name", track["id"])
			artists = ", ".join(a["name"] for a in track.get("artists", []))
			dur = track.get("duration_ms")
			if dur:
				dur /= 1000
			temp = cdict(
				name=name,
				url="ytsearch:" + "".join(c if c.isascii() and c != ":" else "_" for c in f"{name} ~ {artists}"),
				id=track["id"],
				duration=dur,
				research=True,
			)
			out.append(temp)
		return out, total

	# Returns a list of formatted queue entries from a YouTube playlist renderer.
	def extract_playlist_items(self, items):
		token = None
		out = deque()
		for part in items:
			try:
				video = part["playlistVideoRenderer"]
			except KeyError:
				try:
					token = part["continuationItemRenderer"]["continuationEndpoint"]["continuationCommand"]["token"]
				except KeyError:
					print(part)
				continue
			v_id = video['videoId']
			try:
				dur = round_min(float(video["lengthSeconds"]))
			except (KeyError, ValueError):
				try:
					dur = time_parse(video["lengthText"]["simpleText"])
				except KeyError:
					dur = None
			try:
				name = video["title"]["runs"][0]["text"]
			except LookupError:
				name = v_id
			temp = cdict(
				name=name,
				url=f"https://www.youtube.com/watch?v={v_id}",
				duration=dur,
				thumbnail=f"https://i.ytimg.com/vi/{v_id}/maxresdefault.jpg",
			)
			out.append(temp)
		return out, token

	# Returns a subsequent page of a youtube playlist from a page token.
	def get_youtube_continuation(self, token, ctx):
		self.youtube_x += 1
		for i in range(3):
			try:
				resp = reqs.post(
					"https://www.youtube.com/youtubei/v1/browse?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
					headers=self.youtube_header(),
					data=orjson.dumps(dict(
						context=ctx,
						continuation=token,
					)),
				)
				if resp.status_code not in range(200, 400):
					raise
				data = resp.json()
				break
			except:
				print_exc()
				if i:
					time.sleep(i)
		items = data["onResponseReceivedActions"][0]["appendContinuationItemsAction"]["continuationItems"]
		return self.extract_playlist_items(items)

	# Generates a playlist continuation token purely from ID and page number.
	def produce_continuation(self, p, i):
		if not isinstance(p, (bytes, bytearray, memoryview)):
			p = str(p).encode("ascii")
		parts = []
		if i == 1:
			parts.append(b"\xe2\xa9\x85\xb2\x02a\x12$VL")
		else:
			parts.append(b"\xe2\xa9\x85\xb2\x02_\x12$VL")
		parts.append(p)
		if i == 1:
			parts.append(b"\x1a\x14")
		else:
			parts.append(b"\x1a\x12")
		import base64
		def leb128(n):
			data = bytearray()
			while n:
				data.append(n & 127)
				n >>= 7
				if n:
					data[-1] |= 128
			if not data:
				data = b"\x00"
			return data
		key = bytes((8, i, 0x7a, (i != 1) + 6)) + b"PT:" + base64.b64encode(b"\x08" + leb128(i * 100)).rstrip(b"=")
		obj = base64.b64encode(key).replace(b"=", b"%3D")
		parts.append(obj)
		parts.append(b"\x9a\x02\x22")
		parts.append(p)
		code = b"".join(parts)
		return base64.b64encode(code).replace(b"=", b"%3D").decode("ascii")

	# Returns a full youtube playlist.
	def get_youtube_playlist(self, p_id):
		self.youtube_x += 1
		resp = reqs.get(f"https://www.youtube.com/playlist?list={p_id}", headers=self.youtube_header()).content
		client = {}
		try:
			ytcfg = resp[resp.index(b"ytcfg.set"):]
			ytcfg = ytcfg[:ytcfg.index(b";")]
			ytcfg = eval(ytcfg.split(b"(", 1)[-1].rsplit(b")", 1)[0], {}, {})[-1] + "&"
			end = "&"
			start = "client.name="
			cname = ytcfg[ytcfg.index(start) + len(start):]
			client["clientName"] = cname[:cname.index(end)]
			start = "client.version="
			cversion = ytcfg[ytcfg.index(start) + len(start):]
			client["clientVersion"] = cversion[:cversion.index(end)]
		except ValueError:
			pass
		client.setdefault("clientName", "WEB")
		client.setdefault("clientVersion", "2.20211019")
		context = dict(client=client)
		try:
			search = b"var ytInitialData = "
			try:
				resp = resp[resp.index(search) + len(search):]
			except ValueError:
				search = b'window["ytInitialData"] = '
				resp = resp[resp.index(search) + len(search):]
			try:
				resp = resp[:resp.index(b';</script><')]
			except ValueError:
				resp = resp[:resp.index(b'window["ytInitialPlayerResponse"] = null;')]
				resp = resp[:resp.rindex(b";")]
			data = orjson.loads(resp)
		except:
			print(resp)
			raise
		count = int(data["sidebar"]["playlistSidebarRenderer"]["items"][0]["playlistSidebarPrimaryInfoRenderer"]["stats"][0]["runs"][0]["text"].replace(",", ""))
		items = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"][0]["playlistVideoListRenderer"]["contents"]
		entries, token = self.extract_playlist_items(items)
		if count > 100:
			futs = deque()
			if not token:
				token = self.produce_continuation(p_id, 1)
			for page in range(1, ceil(count / 100)):
				futs.append(create_future_ex(self.get_youtube_continuation, token, context))
				token = self.produce_continuation(p_id, page + 1)
			for fut in futs:
				entries.extend(fut.result()[0])
		out = deque()
		urls = set()
		for entry in entries:
			if entry.url not in urls:
				urls.add(entry.url)
				out.append(entry)
		return out

	soundcloud_token = "7g7gIkrcAS05cJVf2FlIsnkOXtg4JdSe"

	def get_soundcloud_playlist(self, url):
		parts = url.split("?", 1)[0].split("/")
		if parts[0] != "https:" or parts[2] not in ("soundcloud.com", "api-v2.soundcloud.com"):
			raise TypeError("Not a SoundCloud playlist.")
		if parts[-1] == "likes":
			return self.get_soundcloud_likes(url)
		api = "https://api-v2.soundcloud.com/"

		resp = requests.get(url, headers=self.youtube_header())
		resp.raise_for_status()
		s = resp.text
		if s[0] == "{" and s[-1] == "}":
			t = resp.json()
			return [cdict(
				name=t["title"],
				url=t["permalink_url"],
				duration=t["duration"] / 1000,
				thumbnail=t["artwork_url"],
			)]
		search = "<script>window.__sc_hydration = "
		s = s[s.index(search) + len(search):]
		s = s[:s.index(";</script>")]
		data = orjson.loads(s)

		emap = {}
		entries = []
		for hydratable in data:
			if hydratable["hydratable"] == "playlist":
				for t in hydratable["data"]["tracks"]:
					try:
						t["title"]
					except KeyError:
						tid = t["id"]
						emap[tid] = len(entries)
						entries.append(None)
					else:
						entry = cdict(
							name=t["title"],
							url=t["permalink_url"],
							duration=t["duration"] / 1000,
							thumbnail=t["artwork_url"],
						)
						entries.append(entry)

		if emap:
			ids = ",".join(map(str, emap))
			url = f"{api}tracks?ids={ids}&client_id={self.soundcloud_token}"
			resp = requests.get(url, headers=self.youtube_header())
			if not resp.content:
				resp.raise_for_status()
			for t, p in zip(resp.json(), emap.values()):
				entry = cdict(
					name=t["title"],
					url=t["permalink_url"],
					duration=t["duration"] / 1000,
					thumbnail=t["artwork_url"],
				)
				entries[p] = entry
		return [e for e in entries if e]

	def get_soundcloud_likes(self, url):
		api = "https://api-v2.soundcloud.com/"
		lim = 1000

		uapi = api + "users/"
		if url.startswith(uapi):
			uid = url[len(uapi):].split("?", 1)[0]
		else:
			resp = requests.get(url, headers=self.youtube_header())
			resp.raise_for_status()
			s = resp.text
			search = 'content="soundcloud://users:'
			s = s[s.index(search) + len(search):]
			uid = s[:s.index('"')]

		futs = []
		entries = []
		while True:
			url = f"{api}users/{uid}/likes?client_id={self.soundcloud_token}&limit={lim}"
			resp = requests.get(url, headers=self.youtube_header())
			if not resp.content:
				resp.raise_for_status()
			data = resp.json()
			for e in data["collection"]:
				try:
					t = e["track"]
				except KeyError:
					p = e["playlist"]
					url = p["permalink_url"]
					if len(futs) >= 12:
						futs.pop(0).result()
					fut = create_future_ex(self.get_soundcloud_playlist, url)
					futs.append(fut)
					entries.append(fut)
				else:
					entry = cdict(
						name=t["title"],
						url=t["permalink_url"],
						duration=t["duration"] / 1000,
						thumbnail=t["artwork_url"],
					)
					entries.append(entry)
			url = data.get("next_href")
			if len(entries) < lim or not url:
				break
			url += f"client_id={self.soundcloud_token}&limit={lim}"

		while True:
			for i, e in enumerate(entries):
				if isinstance(e, concurrent.futures.Future):
					entries = entries[:i] + e.result() + entries[i + 1:]
					break
			else:
				break
		return entries

	def ydl_errors(self, s):
		return "this video has been removed" not in s and "private video" not in s and "has been terminated" not in s

	blocked_yt = False

	# Repeatedly makes calls to youtube-dl until there is no more data to be collected.
	def extract_true(self, url):
		while not is_url(url):
			try:
				resp = self.search_yt(regexp("ytsearch[0-9]*:").sub("", url, 1))[0]
			except:
				resp = self.extract_from(url)
			if "entries" in resp:
				resp = next(iter(resp["entries"]))
			if "duration" in resp and "formats" in resp:
				out = cdict(
					name=resp["title"],
					url=resp["webpage_url"],
					duration=resp["duration"],
					stream=get_best_audio(resp),
					icon=get_best_icon(resp),
					video=get_best_video(resp),
				)
				stream = out.stream
				if "googlevideo" in stream[:64]:
					durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
					if durstr:
						out.duration = round_min(durstr[0])
				return out
			try:
				url = resp["webpage_url"]
			except KeyError:
				try:
					url = resp["url"]
				except KeyError:
					url = resp["id"]
		if is_discord_url(url):
			title = url.split("?", 1)[0].rsplit("/", 1)[-1]
			if "." in title:
				title = title[:title.rindex(".")]
			return dict(url=url, name=title, direct=True)
		try:
			if self.blocked_yt > utc():
				raise PermissionError
			self.youtube_dl_x += 1
			entries = self.downloader.extract_info(url, download=False, process=True)
		except Exception as ex:
			s = str(ex).casefold()
			if type(ex) is not youtube_dl.DownloadError or self.ydl_errors(s):
				if "429" in s:
					self.blocked_yt = utc() + 3600
				try:
					entries = self.extract_backup(url)
				except youtube_dl.DownloadError:
					raise FileNotFoundError("Unable to fetch audio data.")
			else:
				raise
		if "entries" in entries:
			entries = entries["entries"]
		else:
			entries = [entries]
		out = deque()
		for entry in entries:
			temp = cdict(
				name=entry["title"],
				url=entry["webpage_url"],
				duration=entry.get("duration"),
				stream=get_best_audio(entry),
				icon=get_best_icon(entry),
				video=get_best_video(entry),
			)
			out.append(temp)
		return out

	# Extracts audio information from a single URL.
	def extract_from(self, url):
		if is_discord_url(url):
			title = url.split("?", 1)[0].rsplit("/", 1)[-1]
			if "." in title:
				title = title[:title.rindex(".")]
			return dict(url=url, webpage_url=url, title=title, direct=True)
		try:
			if self.blocked_yt > utc():
				raise PermissionError
			if url.startswith("https://www.youtube.com/search") or url.startswith("https://www.youtube.com/results"):
				url = url.split("=", 1)[1].split("&", 1)[0]
			self.youtube_dl_x += 1
			return self.downloader.extract_info(url, download=False, process=False)
		except Exception as ex:
			s = str(ex).casefold()
			if type(ex) is not youtube_dl.DownloadError or self.ydl_errors(s):
				if "429" in s:
					self.blocked_yt = utc() + 3600
				if is_url(url):
					try:
						return self.extract_backup(url)
					except youtube_dl.DownloadError:
						raise FileNotFoundError("Unable to fetch audio data.")
			raise

	# Extracts info from a URL or search, adjusting accordingly.
	def extract_info(self, item, count=1, search=False, mode=None):
		if mode or search and item[:9] not in ("ytsearch:", "scsearch:", "scsearch:", "bcsearch:") and not is_url(item):
			if count == 1:
				c = ""
			else:
				c = count
			item = item.replace(":", "-")
			if mode:
				self.youtube_dl_x += 1
				return self.downloader.extract_info(f"{mode}search{c}:{item}", download=False, process=False)
			exc = ""
			try:
				self.youtube_dl_x += 1
				return self.downloader.extract_info(f"ytsearch{c}:{item}", download=False, process=False)
			except Exception as ex:
				exc = repr(ex)
			try:
				self.youtube_dl_x += 1
				return self.downloader.extract_info(f"scsearch{c}:{item}", download=False, process=False)
			except Exception as ex:
				raise ConnectionError(exc + repr(ex))
		if is_url(item) or not search:
			return self.extract_from(item)
		if item[:9] == "spsearch:":
			query = "https://api.spotify.com/v1/search?type=track%2Cshow_audio%2Cepisode_audio&include_external=audio&limit=1&q=" + url_parse(item[9:])
			resp = reqs.get(query, headers=self.spotify_header).json()
			try:
				track = resp["tracks"]["items"][0]
				name = track.get("name", track["id"])
				artists = ", ".join(a["name"] for a in track.get("artists", ()))
			except LookupError:
				print(resp)
				raise LookupError(f"No results found for {item[9:]}.")
			else:
				item = "ytsearch:" + "".join(c if c.isascii() and c != ":" else "_" for c in f"{name} ~ {artists}")
				self.spotify_x += 1
		elif item[:9] == "bcsearch:":
			query = "https://bandcamp.com/search?q=" + url_parse(item[9:])
			resp = reqs.get(query, headers=self.spotify_header).content
			try:
				resp = resp.split(b'<ul class="result-items">', 1)[1]
				tracks = resp.split(b"<!-- search result type=")
				result = cdict()
				for track in tracks:
					if track.startswith(b"track id=") or track.startswith(b"album id=") and not result:
						ttype = track[:5]
						try:
							track = track.split(b'<img src="', 1)[1]
							result.thumbnail = track[:track.index(b'">')].decode("utf-8", "replace")
						except ValueError:
							pass
						track = track.split(b'<div class="heading">', 1)[1]
						result.title = track.split(b">", 1)[1].split(b"<", 1)[0].strip().decode("utf-8", "replace")
						result.url = track.split(b'href="', 1)[1].split(b'"', 1)[0].split(b"?", 1)[0].decode("utf-8", "replace")
						if ttype == b"track":
							break
				if not result:
					raise LookupError
				print(result)
				return result
			except (LookupError, ValueError):
				print(resp)
				raise LookupError(f"No results found for {item[9:]}.")
			else:
				item = "ytsearch:" + "".join(c if c.isascii() and c != ":" else "_" for c in f"{name} ~ {artists}")
				self.other_x += 1
		self.youtube_dl_x += 1
		return self.downloader.extract_info(item, download=False, process=False)

	# Main extract function, able to extract from youtube playlists much faster than youtube-dl using youtube API, as well as ability to follow spotify links.
	def extract(self, item, force=False, count=1, mode=None, search=True):
		page = None
		output = deque()
		if "youtube.com" in item or "youtu.be/" in item:
			p_id = None
			for x in ("?list=", "&list="):
				if x in item:
					p_id = item[item.index(x) + len(x):]
					p_id = p_id.split("&", 1)[0]
					break
			if p_id:
				try:
					output.extend(self.get_youtube_playlist(p_id))
					# Scroll to highlighted entry if possible
					v_id = None
					for x in ("?v=", "&v="):
						if x in item:
							v_id = item[item.index(x) + len(x):]
							v_id = v_id.split("&", 1)[0]
							break
					if v_id:
						for i, e in enumerate(output):
							if v_id in e.url:
								output.rotate(-i)
								break
					return output
				except:
					output.clear()
					print_exc()
				try:
					entries = list(map(cdict, reqs.get("http://i.mizabot.xyz/ytdl?q=" + item).json()))
					if not entries:
						raise IndexError
				except:
					print_exc()
				else:
					return entries
		elif regexp("^https:\\/\\/soundcloud\\.com\\/[A-Za-z0-9]+\\/sets\\/").search(item) or regexp("^https:\\/\\/soundcloud\\.com\\/[A-Za-z0-9]+\\/likes").search(item) or regexp("^https:\\/\\/api-v2\\.soundcloud\\.com\\/users\\/[0-9]+\\/likes").search(item):
				try:
					return self.get_soundcloud_playlist(item)
				except:
					print_exc()
		elif regexp("(play|open|api)\\.spotify\\.com").search(item):
			# Spotify playlist searches contain up to 100 items each
			if "playlist" in item:
				url = item[item.index("playlist"):]
				url = url[url.index("/") + 1:]
				key = url.split("/", 1)[0]
				url = f"https://api.spotify.com/v1/playlists/{key}/tracks?type=track,episode"
				page = 100
			# Spotify album searches contain up to 50 items each
			elif "album" in item:
				url = item[item.index("album"):]
				url = url[url.index("/") + 1:]
				key = url.split("/", 1)[0]
				url = f"https://api.spotify.com/v1/albums/{key}/tracks?type=track,episode"
				page = 50
			# Single track links also supported
			elif "track" in item:
				url = item[item.index("track"):]
				url = url[url.index("/") + 1:]
				key = url.split("/", 1)[0]
				url = f"https://api.spotify.com/v1/tracks/{key}"
				page = 1
			# Single episode links also supported
			elif "episode" in item:
				url = item[item.index("episode"):]
				url = url[url.index("/") + 1:]
				key = url.split("/", 1)[0]
				url = f"https://api.spotify.com/v1/episodes/{key}"
				page = 1
			else:
				raise TypeError("Unsupported Spotify URL.")
			if page == 1:
				output.extend(self.get_spotify_part(url)[0])
			else:
				futs = deque()
				maxitems = 10000
				# Optimized searching with lookaheads
				for i, curr in enumerate(range(0, maxitems, page)):
					with delay(0.03125):
						if curr >= maxitems:
							break
						search = f"{url}&offset={curr}&limit={page}"
						fut = create_future_ex(self.get_spotify_part, search)
						print("Sent 1 spotify search.")
						futs.append(fut)
						if not (i < 1 or math.log2(i + 1) % 1) or not i & 7:
							while futs:
								fut = futs.popleft()
								res = fut.result()
								if not i:
									maxitems = res[1] + page
								if not res[0]:
									maxitems = 0
									futs.clear()
									break
								output += res[0]
				while futs:
					output.extend(futs.popleft().result()[0])
				# Scroll to highlighted entry if possible
				v_id = None
				for x in ("?highlight=spotify:track:", "&highlight=spotify:track:"):
					if x in item:
						v_id = item[item.index(x) + len(x):]
						v_id = v_id.split("&", 1)[0]
						break
				if v_id:
					for i, e in enumerate(output):
						if v_id == e.get("id"):
							output.rotate(-i)
							break
		if is_url(item) and tuple(reversed(item.split("/", 3)[2].encode("utf-8"))) == (109, 111, 99, 46, 117, 109, 105, 110, 97, 120):
			b = reqs.get(item, headers=self.youtube_header()).content
			search = b'itemprop="name" content="'
			try:
				i = b.index(search)
			except ValueError:
				pass
			else:
				s, b = b[i + len(search):].split(b'"', 1)
				name = s.decode("utf-8", "replace")
				search = b'itemprop="duration" content="'
				s, b = b.split(search, 1)[-1].split(b'"', 1)
				duration = time_parse(s.decode("utf-8", "replace").lstrip("P").rstrip("S").replace("DT", ":").replace("H", ":").replace("M", ":"))
				search = b'itemprop="thumbnailUrl" content="'
				s, b = b.split(search, 1)[-1].split(b'"', 1)
				thumbnail = s.decode("utf-8", "replace")
				search = b'itemprop="embedURL" content="'
				with requests.head(b.split(search, 1)[-1].split(b'"', 1)[0].decode("utf-8", "replace"), headers=self.youtube_header(), allow_redirects=True) as resp:
					stream = resp.url
				temp = cdict(
					name=name,
					url=item,
					duration=duration,
					stream=stream,
					icon=thumbnail,
					video=stream,
				)
				output.append(temp)
		# Only proceed if no items have already been found (from playlists in this case)
		if not len(output):
			# Allow loading of files output by ~dump
			if is_url(item):
				url = verify_url(item)
				if url[-5:] == ".json" or url[-4:] in (".txt", ".bin", ".zip"):
					s = reqs.get(url).content
					if not s.startswith(b"{"):
						b = io.BytesIO(s)
						with zipfile.ZipFile(data, allowZip64=True, strict_timestamps=False) as z:
							s = z.read(z.namelist()[0])
					d = orjson.loads(s)
					q = d["queue"][:262144]
					return [cdict(name=e["name"], url=e["url"], duration=e.get("duration")) for e in q]
			elif mode in (None, "yt"):
				with suppress(NotImplementedError):
					return self.search_yt(item)[:count]
			# Otherwise call automatic extract_info function
			resp = self.extract_info(item, count, search=search, mode=mode)
			if resp.get("_type") == "url":
				resp = self.extract_from(resp["url"])
			if resp is None or not len(resp):
				raise LookupError(f"No results for {item}")
			# Check if result is a playlist
			if resp.get("_type") == "playlist":
				entries = resp["entries"]
				if force:
					for entry in entries:
						# Extract full data if playlist only contains 1 item
						data = self.extract_from(entry["url"])
						temp = {
							"name": data["title"],
							"url": data["webpage_url"],
							"duration": float(data["duration"]),
							"stream": get_best_audio(resp),
							"icon": get_best_icon(resp),
							"video": get_best_video(resp),
						}
						output.append(cdict(temp))
				else:
					for i, entry in enumerate(entries):
						if not i:
							# Extract full data from first item only
							temp = self.extract(entry["url"], search=False)[0]
						else:
							# Get as much data as possible from all other items, set "research" flag to have bot lazily extract more info in background
							try:
								found = True
								if "title" in entry:
									title = entry["title"]
								else:
									title = entry["url"].rsplit("/", 1)[-1]
									if "." in title:
										title = title[:title.rindex(".")]
									found = False
								if entry.get("duration") is not None:
									dur = float(entry["duration"])
								else:
									dur = None
								url = entry.get("webpage_url", entry.get("url", entry.get("id")))
								if not url:
									continue
								temp = {
									"name": title,
									"url": url,
									"duration": dur,
								}
								if not is_url(url):
									if entry.get("ie_key", "").casefold() == "youtube":
										temp["url"] = f"https://www.youtube.com/watch?v={url}"
								temp["research"] = True
							except:
								print_exc()
						output.append(cdict(temp))
			else:
				# Single item results must contain full data, we take advantage of that here
				found = "duration" in resp
				if found:
					dur = resp["duration"]
				else:
					dur = None
				temp = {
					"name": resp.get("title") or resp["webpage_url"].rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0],
					"url": resp.get("webpage_url") or resp["url"],
					"duration": dur,
					"stream": get_best_audio(resp),
					"icon": get_best_icon(resp),
					"video": get_best_video(resp),
				}
				output.append(cdict(temp))
		return output

	def item_yt(self, item):
		video = next(iter(item.values()))
		if "videoId" not in video:
			return
		try:
			dur = time_parse(video["lengthText"]["simpleText"])
		except KeyError:
			dur = None
		try:
			title = video["title"]["runs"][0]["text"]
		except KeyError:
			title = video["title"]["simpleText"]
		try:
			tn = video["thumbnail"]
		except KeyError:
			thumbnail = None
		else:
			if type(tn) is dict:
				thumbnail = sorted(tn["thumbnails"], key=lambda t: t.get("width", 0) * t.get("height", 0))[-1]["url"]
			else:
				thumbnail = tn
		try:
			views = int(video["viewCountText"]["simpleText"].replace(",", "").replace("views", "").replace(" ", ""))
		except (KeyError, ValueError):
			views = 0
		return cdict(
			name=video["title"]["runs"][0]["text"],
			url=f"https://www.youtube.com/watch?v={video['videoId']}",
			duration=dur,
			icon=thumbnail,
			views=views,
		)

	def parse_yt(self, s):
		data = orjson.loads(s)
		results = deque()
		try:
			pages = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
		except KeyError:
			pages = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"]
		for page in pages:
			try:
				items = next(iter(page.values()))["contents"]
			except KeyError:
				continue
			for item in items:
				if "promoted" not in next(iter(item)).casefold():
					entry = self.item_yt(item)
					if entry is not None:
						results.append(entry)
		return sorted(results, key=lambda entry: entry.views, reverse=True)

	def search_yt(self, query, skip=False):
		out = None
		if not skip:
			try:
				resp = self.extract_info(query)
				if resp.get("_type", None) == "url":
					resp = self.extract_from(resp["url"])
				if resp.get("_type", None) == "playlist":
					entries = list(resp["entries"])
				else:
					entries = [resp]
				out = []
				for entry in entries:
					try:
						found = True
						if "title" in entry:
							title = entry["title"]
						else:
							title = entry["url"].rsplit("/", 1)[-1]
							if "." in title:
								title = title[:title.rindex(".")]
							found = False
						if "duration" in entry:
							dur = float(entry["duration"])
						else:
							dur = None
						url = entry.get("webpage_url", entry.get("url", entry.get("id")))
						if not url:
							continue
						temp = cdict(name=title, url=url, duration=dur)
						if not is_url(url):
							if entry.get("ie_key", "").casefold() == "youtube":
								temp["url"] = f"https://www.youtube.com/watch?v={url}"
						temp["research"] = True
						out.append(temp)
					except:
						print_exc()
			except:
				print_exc()
		if not out:
			url = f"https://www.youtube.com/results?search_query={verify_url(query)}"
			self.youtube_x += 1
			resp = reqs.get(url, headers=self.youtube_header()).content
			result = None
			s = resp
			with suppress(ValueError):
				with suppress(ValueError):
					s = s[s.index(b"// scraper_data_begin") + 21:s.rindex(b"// scraper_data_end")]
				s = s[s.index(b"var ytInitialData = ") + 20:]
				s = s[:s.index(b";</script>")]
				result = self.parse_yt(s)
			with suppress(ValueError):
				s = s[s.index(b'window["ytInitialData"] = ') + 26:]
				s = s[:s.index(b'window["ytInitialPlayerResponse"] = null;')]
				s = s[:s.rindex(b";")]
				result = self.parse_yt(s)
			if result is None:
				raise NotImplementedError("Unable to read json response.")
			q = to_alphanumeric(full_prune(query))
			high = deque()
			low = deque()
			for entry in result:
				if entry.duration:
					name = full_prune(entry.name)
					aname = to_alphanumeric(name)
					spl = aname.split()
					if entry.duration < 960 or "extended" in q or "hour" in q or "extended" not in spl and "hour" not in spl and "hours" not in spl:
						if fuzzy_substring(aname, q, match_length=False) >= 0.5:
							high.append(entry)
							continue
				low.append(entry)
			def key(entry):
				coeff = fuzzy_substring(to_alphanumeric(full_prune(entry.name)), q, match_length=False)
				if coeff < 0.5:
					coeff = 0
				return coeff
			out = sorted(high, key=key, reverse=True)
			out.extend(sorted(low, key=key, reverse=True))
		return out

	# Performs a search, storing and using cached search results for efficiency.
	def search(self, item, force=False, mode=None, count=1):
		item = verify_search(item)
		if mode is None and count == 1 and item in self.searched:
			if utc() - self.searched[item].t < 60:
				return self.searched[item].data
			else:
				self.searched.pop(item)
		while len(self.searched) > 262144:
			self.searched.pop(next(iter(self.searched)))
		obj = cdict(t=utc())
		obj.data = output = self.extract(item, force, mode=mode, count=count)
		if obj.data:
			self.searched[item] = obj
		return output

	# Gets the stream URL of a queue entry, `download` being a redundant variable kept for compatibility.
	def get_stream(self, entry, force=False, download=True):
		video = entry.get("video", None)
		stream = entry.get("stream", None)
		icon = entry.get("icon", None)
		# Use SHA-256 hash of URL to avoid filename conflicts
		h = shash(entry["url"])
		fn = h + ".opus"
		# "none" indicates stream is currently loading
		if stream == "none" and not force:
			return None
		entry["stream"] = "none"
		searched = False
		name = entry.get("name")
		# If "research" tag is set, entry does not contain full data and requires another search
		if "research" in entry:
			try:
				self.extract_single(entry)
				searched = True
				entry.pop("research", None)
			except:
				print_exc()
				entry.pop("research", None)
				raise
			else:
				name = entry["name"]
				stream = entry.get("stream")
				icon = entry.get("icon")
				video = entry.get("video")
		# If stream is still not found or is a soundcloud audio fragment playlist file, perform secondary youtube-dl search
		if stream in (None, "none"):
			data = self.search(entry["url"])
			name = data[0]["name"]
			stream = data[0].get("stream") or data[0]["url"]
			icon = data[0].setdefault("icon", data[0]["url"])
			video = data[0].setdefault("video", data[0]["url"])
			entry["duration"] = data[0].get("duration")
		if not searched and (not is_url(stream) or not isfinite(float(entry.get("duration") or inf)) or stream.startswith("ytsearch:") or stream.startswith("https://cf-hls-media.sndcdn.com/") or expired(stream)):
			data = self.extract(entry["url"])
			name = data[0]["name"]
			stream = data[0].get("stream") or data[0]["url"]
			icon = data[0].setdefault("icon", data[0]["url"])
			video = data[0].setdefault("video", data[0]["url"])
			entry.update(data[0])
		entry["name"] = name
		entry["stream"] = stream
		entry["icon"] = icon
		entry["video"] = video
		if "googlevideo" in stream[:64]:
			durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
			if durstr:
				entry["duration"] = round_min(durstr[0])
		if entry["url"] not in self.searched:
			self.searched[entry["url"]] = cdict(
				t=utc(),
				data=[cdict(entry)],
			)
		else:
			with suppress(KeyError):
				self.searched[entry["url"]]["duration"] = entry["duration"]
		self.searched[entry["url"]].data[0].update(entry)
		return stream

	# Extracts full data for a single entry. Uses cached results for optimization.
	def extract_single(self, i, force=False):
		item = i.url
		if not force:
			if item in self.searched and not item.startswith("ytsearch:"):
				if utc() - self.searched[item].t < 60:
					it = self.searched[item].data[0]
					i.update(it)
					if i.get("stream") not in (None, "none"):
						return True
				else:
					self.searched.pop(item, None)
			while len(self.searched) > 262144:
				self.searched.pop(next(iter(self.searched)))
		data = self.extract_true(item)
		if "entries" in data:
			data = data["entries"][0]
		elif not issubclass(type(data), collections.abc.Mapping):
			data = data[0]
		if data.get("research"):
			data = self.extract_true(data["url"])[0]
		elif not issubclass(type(data), collections.abc.Mapping):
			data = data[0]
		obj = cdict(t=utc())
		obj.data = out = [cdict(
			name=data.get("title") or data.get("name"),
			url=data.get("webpage_url") or data.get("url"),
			stream=data.get("stream") or get_best_audio(data),
			icon=data.get("icon") or get_best_icon(data),
			video=data.get("video") or get_best_video(data),
		)]
		try:
			out[0].duration = data["duration"]
		except KeyError:
			out[0].research = True
		self.searched[item] = obj
		it = out[0]
		i.update(it)
		return True

mmap = {
	"“": '"',
	"”": '"',
	"„": '"',
	"‘": "'",
	"’": "'",
	"‚": "'",
	"〝": '"',
	"〞": '"',
	"⸌": "'",
	"⸍": "'",
	"⸢": "'",
	"⸣": "'",
	"⸤": "'",
	"⸥": "'",
	"⸨": "((",
	"⸩": "))",
	"⟦": "[",
	"⟧": "]",
	"〚": "[",
	"〛": "]",
	"「": "[",
	"」": "]",
	"『": "[",
	"』": "]",
	"【": "[",
	"】": "]",
	"〖": "[",
	"〗": "]",
	"（": "(",
	"）": ")",
	"［": "[",
	"］": "]",
	"｛": "{",
	"｝": "}",
	"⌈": "[",
	"⌉": "]",
	"⌊": "[",
	"⌋": "]",
	"⦋": "[",
	"⦌": "]",
	"⦍": "[",
	"⦐": "]",
	"⦏": "[",
	"⦎": "]",
	"⁅": "[",
	"⁆": "]",
	"〔": "[",
	"〕": "]",
	"«": "<<",
	"»": ">>",
	"❮": "<",
	"❯": ">",
	"❰": "<",
	"❱": ">",
	"❬": "<",
	"❭": ">",
	"＜": "<",
	"＞": ">",
	"⟨": "<",
	"⟩": ">",
}
mtrans = "".maketrans(mmap)

lyric_trans = re.compile(
	(
		"[([]+"
		"(((official|full|demo|original|extended) *)?"
		"((version|ver.?) *)?"
		"((w\\/)?"
		"(lyrics?|vocals?|music|ost|instrumental|acoustic|studio|hd|hq|english) *)?"
		"((album|video|audio|cover|remix) *)?"
		"(upload|reupload|version|ver.?)?"
		"|(feat|ft)"
		".+)"
		"[)\\]]+"
	),
	flags=re.I,
)

def extract_lyrics(s):
	s = s[s.index("__ = JSON.parse(") + len("__ = JSON.parse("):]
	s = s[:s.index("</script>")]
	try:
		s = s[:s.index("window.__")]
	except ValueError:
		pass
	s = s[:s.rindex(");")]
	data = eval(s, {}, {})
	d = eval_json(data)
	lyrics = d["songPage"]["lyricsData"]["body"]["children"][0]["children"]
	newline = True
	output = ""
	while lyrics:
		line = lyrics.pop(0)
		if type(line) is str:
			if line:
				if line.startswith("["):
					output += "\n"
					newline = False
				if "]" in line:
					if line == "]":
						if output.endswith(" ") or output.endswith("\n"):
							output = output[:-1]
					newline = True
				output += line + ("\n" if newline else (" " if not line.endswith(" ") else ""))
		elif type(line) is dict:
			if "children" in line:
				lyrics = line["children"] + lyrics
	return output

def get_lyrics(search):
	search = search.translate(mtrans)
	item = verify_search(to_alphanumeric(lyric_trans.sub("", search)))
	ic = item.casefold()
	if ic.endswith(" with lyrics"):
		item = item[:-len(" with lyrics")]
	elif ic.endswith(" lyrics"):
		item = item[:-len(" lyrics")]
	elif ic.endswith(" acoustic"):
		item = item[:-len(" acoustic")]
	item = item.rsplit(" ft ", 1)[0].strip()
	if not item:
		item = verify_search(to_alphanumeric(search))
		if not item:
			item = search
	url = f"https://genius.com/api/search/multi?q={item}"
	for i in range(2):
		header = {"User-Agent": "Mozilla/6.0"}
		data = {"q": item}
		rdata = reqs.get(url, data=data, headers=header, timeout=18).json()
		hits = itertools.chain(*(sect["hits"] for sect in rdata["response"]["sections"]))
		name = None
		path = None
		for h in hits:
			try:
				name = h["result"]["title"]
				path = h["result"]["api_path"]
				break
			except KeyError:
				pass
		if path and name:
			s = "https://genius.com" + path
			page = reqs.get(s, headers=header)
			text = page.text
			if "BeautifulSoup" not in globals():
				bs4 = __import__("bs4")
				globals()["BeautifulSoup"] = bs4.BeautifulSoup
			html = BeautifulSoup(text, "html.parser")
			lyricobj = html.find('div', class_='lyrics')
			if lyricobj is not None:
				lyrics = lyricobj.get_text().strip().replace("\r", "")
				return name, lyrics
			try:
				lyrics = extract_lyrics(text).strip().replace("\r", "")
				return name, lyrics
			except:
				if i:
					raise
	raise LookupError(f"No results for {item}.")