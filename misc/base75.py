"""Base75 string encoding: a middle ground between base64 and base85.

By using a safer character set, the resulting encoded strings may be applied in many cases where base85 (and its variants) would fail, such as JSON strings, URIs, and URLs in markdown text, all while remaining more efficient than base64.

- The provided charset (also in ASCII order) is !$*,-.0123456789:;=@ABCDEFGHIJKLMNOPQRSTUVWXYZ^_abcdefghijklmnopqrstuvwxyz~
- Each 7 bytes of data is represented with 9 ascii characters.
- This is possible as 75^9 = 75084686279296875 >= 256^7 = 72057594037927936.
- Padding is optional, done using a "+" character, which is technically unsafe in some contexts (e.g. URL-encoding may convert to " ") but can safely be stripped away without impacting the decoded result, so long as it is confined to the end of strings.
- Non-canonical encodings, as well as any invalid character outside the provided set appearing within the string (including padding not correctly placed at the end), should result in the decode operation being rejected.
- Note that some characters in the set (e.g. $*,;:=@) are RFC 3986 sub-delims and the ^ character (also used) may be reserved in some scenarios, which systems utilising base75 must take into account.

Theoretical efficiency comparison:
```
+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| base16 | base32 | base36 | base41 | base45 | base64 | base75 | base85 |  ascii |
+--------+--------+--------+--------+--------+--------+--------+--------+--------+
|   1/2  |   5/8  |  8/13  |   2/3  |   2/3  |   3/4  |   7/9  |   4/5  |   7/8  |
|   50%  |  62.5% |  61.5% |  66.7% |  66.7% |   75%  |  77.8% |   80%  |  87.5% |
+--------+--------+--------+--------+--------+--------+--------+--------+--------+
```
"""

import numpy as np

ALPHABET = "!$*,-.0123456789:;=@ABCDEFGHIJKLMNOPQRSTUVWXYZ^_abcdefghijklmnopqrstuvwxyz~"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}
PAD = "+"
# for tail encoding/decoding
BYTES_TO_CHARS = {1: 2, 2: 3, 3: 4, 4: 6, 5: 7, 6: 8}
CHARS_TO_BYTES = {v: k for k, v in BYTES_TO_CHARS.items()}

# --- NumPy lookup tables (built once at import) ---
_ENCODE_LUT = np.frombuffer(ALPHABET.encode("ascii"), dtype=np.uint8)
_DECODE_LUT = np.full(256, 256, dtype=np.uint64)
for _c, _i in _INDEX.items():
	_DECODE_LUT[ord(_c)] = _i

_POW75 = np.uint64(75) ** np.arange(9, dtype=np.uint64)[::-1]
_POW256 = np.uint64(256) ** np.arange(7, dtype=np.uint64)[::-1]


def _scalar_encode_tail(chunk: bytes) -> str:
	"""Encode a trailing partial group (1-6 bytes) without padding."""
	n = BYTES_TO_CHARS[len(chunk)]
	v = int.from_bytes(chunk, "big")
	digits = [0] * n
	for j in range(n - 1, -1, -1):
		v, digits[j] = divmod(v, 75)
	return "".join(ALPHABET[d] for d in digits)

def encode(data: bytes, pad: bool = False) -> str:
	"""Encode bytes to a base75 string.

	If pad is True, a short trailing group is right-padded with "+" so the
	output length is a multiple of 9.
	"""
	if not data:
		return ""
	n_full, rem = divmod(len(data), 7)
	parts = []
	if n_full:
		arr = np.frombuffer(data[: n_full * 7], dtype=np.uint8).reshape(n_full, 7)
		v = (arr.astype(np.uint64) * _POW256).sum(axis=1)
		# Vectorized base-75 decomposition, most significant digit first
		digits = (v[:, None] // _POW75[None, :]) % np.uint64(75)
		parts.append(_ENCODE_LUT[digits].tobytes().decode("ascii"))
	if rem:
		tail = _scalar_encode_tail(data[n_full * 7:])
		if pad:
			tail += PAD * (9 - len(tail))
		parts.append(tail)
	return "".join(parts)


def _scalar_decode_tail(body: str, n_bytes: int) -> bytes:
	"""Decode a trailing partial group (length already validated)."""
	v = 0
	for c in body:
		if c not in _INDEX:
			raise ValueError(f"invalid character {c!r}")
		v = v * 75 + _INDEX[c]
	if v >= 1 << (8 * n_bytes):
		raise ValueError("non-canonical encoding")
	return v.to_bytes(n_bytes, "big")

def decode(s: str) -> bytes:
	"""Decode a base75 string back to bytes.

	Accepts both unpadded input and input right-padded with "+" (or " ", which
	is what "+" becomes under URL-decoding). When padding is present the total
	length must be a multiple of 9 and the padding may only fill out a genuinely
	short final group; padding appended to an already-complete group is rejected.
	"""
	body = s.rstrip(PAD + " ")
	if len(body) != len(s):
		# Padding was present, so it must fill a short trailing group exactly.
		if len(s) % 9:
			raise ValueError("padded length is not a multiple of 9")
		if len(body) % 9 not in CHARS_TO_BYTES:
			raise ValueError("misplaced padding")
	if not body:
		return b""

	# Full 9-char data groups, plus an optional short tail group
	n_full, tail_len = divmod(len(body), 9)
	if tail_len and tail_len not in CHARS_TO_BYTES:
		raise ValueError("invalid group length")

	out = b""
	if n_full:
		full_str = body[: n_full * 9]
		try:
			raw = np.frombuffer(full_str.encode("ascii"), dtype=np.uint8)
		except UnicodeEncodeError as e:
			raise ValueError(f"invalid character {full_str[e.start]!r}") from None
		idx = _DECODE_LUT[raw]
		bad = idx >= 256
		if bad.any():
			raise ValueError(f"invalid character {full_str[int(np.argmax(bad))]!r}")
		v = (idx.reshape(n_full, 9) * _POW75[None, :]).sum(axis=1)
		if (v >= np.uint64(1 << 56)).any():
			raise ValueError("non-canonical encoding")
		# Convert each resulting 8 byte group to 7 bytes
		out = v.astype(">u8").view(np.uint8).reshape(n_full, 8)[:, 1:].tobytes()
	if tail_len:
		out += _scalar_decode_tail(body[n_full * 9:], CHARS_TO_BYTES[tail_len])
	return out