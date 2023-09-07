import os, sys
if __name__ == "__main__":
	if "-g" in sys.argv:
		i = sys.argv.index("-g")
		sys.argv.pop(i)
		device = int(sys.argv.pop(i))
		os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
		device = 0
	else:
		device = None
	if len(sys.argv) < 3 or "-d" not in sys.argv and "-e" not in sys.argv and "-i" not in sys.argv:
		raise SystemExit(f"Usage: {sys.executable} {' '.join(sys.argv)} <-e | -d> <ecdc-file-or-url>")
	if "-d" in sys.argv:
		mode = "decode"
		sys.argv.remove("-d")
		fn = sys.argv[-1]
		is_url = lambda url: "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")
		if is_url(fn):
			import urllib.request, random
			def header():
				return {
					"User-Agent": f"Mozilla/5.{random.randint(1, 9)} (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
					"DNT": "1",
					"X-Forwarded-For": ".".join(str(random.randint(0, 255)) for _ in range(4)),
				}
			req = urllib.request.Request(fn, headers=header())
			file = urllib.request.urlopen(req)
		else:
			file = open(fn, "rb")
	elif "-i" in sys.argv:
		mode = "info"
		sys.argv.remove("-i")
		fn = sys.argv[-1]
		is_url = lambda url: "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")
		if is_url(fn):
			import urllib.request, random
			def header():
				return {
					"User-Agent": f"Mozilla/5.{random.randint(1, 9)} (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
					"DNT": "1",
					"X-Forwarded-For": ".".join(str(random.randint(0, 255)) for _ in range(4)),
				}
			req = urllib.request.Request(fn, headers=header())
			file = urllib.request.urlopen(req)
		else:
			file = open(fn, "rb")
		from encodec import binary
		fo = file
		header_bytes = binary._read_exactly(fo, binary._encodec_header_struct.size)
		magic, version, meta_size = binary._encodec_header_struct.unpack(header_bytes)
		if magic != binary._ENCODEC_MAGIC:
			raise ValueError("File is not in ECDC format.")
		print("Version:", version)
		meta_bytes = binary._read_exactly(fo, meta_size)
		try:
			import orjson
		except ImportError:
			import json
			metadata = json.loads(meta_bytes.decode('utf-8'))
		else:
			metadata = orjson.loads(meta_bytes)
		if "n" in metadata:
			print("Name:", metadata.pop("n"))
		if "al" in metadata:
			print("Duration:", metadata["al"] / float(metadata.get("m", "_48").rsplit("_", 1)[-1].removesuffix("khz")) / 1000 / 2)
		for k, v in metadata.items():
			print(f"{k.upper()}:", v)
		raise SystemExit
	else:
		mode = "encode"
		sys.argv.remove("-e")
		if "-n" in sys.argv:
			i = sys.argv.index("-n")
			sys.argv.pop(i)
			name = sys.argv.pop(i)
			if name.startswith(" "):
				import base64
				name = base64.b64decode(name.encode("ascii") + b"==").decode("utf-8", "replace")
		else:
			name = None
		if "-s" in sys.argv:
			i = sys.argv.index("-s")
			sys.argv.pop(i)
			source = sys.argv.pop(i)
		else:
			source = None
		if "-b" in sys.argv:
			i = sys.argv.index("-b")
			sys.argv.pop(i)
			bitrate = float(sys.argv.pop(i).removesuffix("k"))
		else:
			bitrate = 24
		fn = sys.argv[-1]
		file = open(fn, "wb")


import io
import math
import struct
import time
import typing as tp
import torch

import encodec
from encodec import binary
from encodec.quantization.ac import ArithmeticCoder, ArithmeticDecoder, build_stable_quantized_cdf
from encodec.model import EncodecModel, EncodedFrame

MODELS = {
	'encodec_24khz': EncodecModel.encodec_model_24khz,
	'encodec_48khz': EncodecModel.encodec_model_48khz,
}


def stream_from_file(fo: tp.IO[bytes], device='cpu') -> tp.Tuple[torch.Tensor, int]:
	"""Stream from a file-object.
	Returns a tuple `(wav, sample_rate)`.

	Args:
		fo (IO[bytes]): file-object from which to read. If you want to decompress
			from `bytes` instead, see `decompress`.
		device: device to use to perform the computations.
	"""
	header_bytes = binary._read_exactly(fo, binary._encodec_header_struct.size)
	magic, version, meta_size = binary._encodec_header_struct.unpack(header_bytes)
	if magic != binary._ENCODEC_MAGIC:
		raise ValueError("File is not in ECDC format.")
	if version not in (0, 192):
		raise ValueError("Version not supported.")
	meta_bytes = binary._read_exactly(fo, meta_size)
	try:
		import orjson
	except ImportError:
		import json
		metadata = json.loads(meta_bytes.decode('utf-8'))
	else:
		metadata = orjson.loads(meta_bytes)
	model_name = metadata['m']
	audio_length = metadata['al']
	num_codebooks = metadata['nc']
	use_lm = metadata['lm']
	assert isinstance(audio_length, int)
	assert isinstance(num_codebooks, int)
	if model_name not in MODELS:
		raise ValueError(f"The audio was compressed with an unsupported model {model_name}.")
	try:
		model = MODELS[model_name]().to(device=device)
	except:
		import traceback
		traceback.print_exc()
		sys.stderr.write(f"MODEL FAIL: {model_name} {device}\n")

	if use_lm:
		lm = model.get_lm_model()

	frames: tp.List[EncodedFrame] = []
	segment_length = model.segment_length or audio_length
	segment_stride = model.segment_stride or audio_length
	i = -1
	for offset in range(0, audio_length, segment_stride):
		# This section is the original ecdc decoder
		this_segment_length = min(audio_length - offset, segment_length)
		frame_length = int(math.ceil(this_segment_length * model.frame_rate / model.sample_rate))
		if model.normalize:
			scale_f, = struct.unpack('!f', binary._read_exactly(fo, struct.calcsize('!f')))
			scale = torch.tensor(scale_f, device=device).view(1)
		else:
			scale = None
		if use_lm:
			decoder = ArithmeticDecoder(fo)
			states: tp.Any = None
			offset = 0
			input_ = torch.zeros(1, num_codebooks, 1, dtype=torch.long, device=device)
		else:
			unpacker = binary.BitUnpacker(model.bits_per_codebook, fo)
		frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long, device=device)
		for t in range(frame_length):
			if use_lm:
				with torch.no_grad():
					probas, states, offset = lm(input_, states, offset)
			code_list: tp.List[int] = []
			for k in range(num_codebooks):
				if use_lm:
					q_cdf = build_stable_quantized_cdf(
						probas[0, :, k, 0], decoder.total_range_bits, check=False)
					code = decoder.pull(q_cdf)
				else:
					code = unpacker.pull()
				if code is None:
					raise EOFError("The stream ended sooner than expected.")
				code_list.append(code)
			codes = torch.tensor(code_list, dtype=torch.long, device=device)
			frame[0, :, t] = codes
			if use_lm:
				input_ = 1 + frame[:, :, t: t + 1]
		frames.append((frame, scale))
		# Problem: Streaming the audio requires the first packet to be read asap, however each packet does not perfectly blend into the next unless both are decoded as one packet, which then does not perfectly blend into the packet after.
		# Possible solution: Read windows of 3 consecutive packets at once, only outputting the central packet at any given time. This will decode the sequence perfectly; however this means a +200% computational overhead due to having to decode 3x the data compared to a single decode on the entire file.
		# Proposed solution: Grab audio packets in sequence of triangle numbers -1; i.e. packets will be gathered and converted on iterations 0, 2, 5, 9, 14 etc. This iteratively expands the window, reducing computational overhead from window overlap later on, while still allowing the first window to be streamed as soon as possible. If the file is long enough, computational overhead approaches 0%.
		if len(frames) >= 2 and (i * 8 + 1) ** 0.5 % 1 == 0 or offset + segment_stride >= audio_length:
			# sys.stderr.write(str(len(frames)) + "\n")
			with torch.no_grad():
				wav = model.decode(frames)
			# Only include first window if on the initial iteration; skip otherwise as it is only used to bridge from the last window
			if len(frames) < 3:
				start = 0
			else:
				start = segment_stride
			# Only include the last window if on the final iteration; skip otherwise as it is only used to bridge to the next window
			if offset + segment_stride >= audio_length:
				end = audio_length
			else:
				end = segment_stride * (len(frames) - 1)
			yield wav[0, :, start:end], model.sample_rate
			frames = [frames[-2], frames[-1]]
		i += 1

def stream_to_file(fo: tp.IO[bytes], use_lm: bool = False, hq: bool = True, bitrate: float = 24, name=None, source=None, device='cpu'):
	"""Compress a waveform to a file-object using the given model.

	Args:
		model (EncodecModel): a pre-trained EncodecModel to use to compress the audio.
		wav (torch.Tensor): waveform to compress, should have a shape `[C, T]`, with `C`
			matching `model.channels`, and the proper sample rate (e.g. `model.sample_rate`).
			Use `utils.convert_audio` if this is not the case.
		fo (IO[bytes]): file-object to which the compressed bits will be written.
			See `compress` if you want obtain a `bytes` object instead.
		use_lm (bool): if True, use a pre-trained language model to further
			compress the stream using Entropy Coding. This will slow down compression
			quite a bit, expect between 20 to 30% of size reduction.
	"""
	hq = hq and bitrate >= 3
	model_name = 'encodec_48khz' if hq else 'encodec_24khz'
	if model_name not in MODELS:
		raise ValueError(f"The provided model {model_name} is not supported.")
	try:
		model = MODELS[model_name]().to(device=device)
	except:
		import traceback
		traceback.print_exc()
		sys.stderr.write(f"MODEL FAIL: {model_name} {device}\n")
	model.set_target_bandwidth(bitrate)

	if use_lm:
		lm = model.get_lm_model()

	dtype = torch.float32# if device in (-1, "cpu") else torch.float16
	data = sys.stdin.buffer.read()
	wav = torch.frombuffer(data, dtype=torch.int16).reshape((len(data) // 4, 2)).T.to(device=device)
	high = 32767 + (-32768 in wav)
	wav = wav.to(dtype)
	wav *= 1 / high
	if not hq:
		wav = wav.mean(-2, keepdim=True)

	with torch.no_grad():
		frames = model.encode(wav[None])

	metadata = {
		'm': model.name,							# model name
		'al': wav.shape[-1],						# audio_length
		'nc': frames[0][0].shape[1],				# num_codebooks
		'lm': use_lm,								# use lm?
	}
	if name:
		metadata["n"] = name
	if source:
		metadata["s"] = source
	try:
		import orjson
	except:
		import json
		meta_dumped = json.dumps(metadata, indent=None, separators=(",", ":")).encode('utf-8')
	else:
		meta_dumped = orjson.dumps(metadata)
	version = 192
	header = binary._encodec_header_struct.pack(binary._ENCODEC_MAGIC, version, len(meta_dumped))
	fo.write(header)
	fo.write(meta_dumped)

	for (frame, scale) in frames:
		if scale is not None:
			fo.write(struct.pack('!f', scale.cpu().item()))
		_, K, T = frame.shape
		if use_lm:
			coder = ArithmeticCoder(fo)
			states: tp.Any = None
			offset = 0
			input_ = torch.zeros(1, K, 1, dtype=torch.long, device=wav.device)
		else:
			packer = binary.BitPacker(model.bits_per_codebook, fo)
		for t in range(T):
			if use_lm:
				with torch.no_grad():
					probas, states, offset = lm(input_, states, offset)
				# We emulate a streaming scenario even though we do not provide an API for it.
				# This gives us a more accurate benchmark.
				input_ = 1 + frame[:, :, t: t + 1]
			for k, value in enumerate(frame[0, :, t].tolist()):
				if use_lm:
					q_cdf = build_stable_quantized_cdf(
						probas[0, :, k, 0], coder.total_range_bits, check=False)
					coder.push(value, q_cdf)
				else:
					packer.push(value)
	if use_lm:
		coder.flush()
	else:
		packer.flush()
	fo.flush()
	fo.close()


if os.path.exists("auth.json"):
	import json
	with open("auth.json", "rb") as f:
		AUTH = json.load(f)
	cachedir = AUTH.get("cache_path") or None
	if cachedir:
		os.environ["HF_HOME"] = f"{cachedir}/huggingface"
		os.environ["TORCH_HOME"] = f"{cachedir}/torch"
		os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cachedir}/huggingface/hub"
		os.environ["TRANSFORMERS_CACHE"] = f"{cachedir}/huggingface/transformers"
		os.environ["HF_DATASETS_CACHE"] = f"{cachedir}/huggingface/datasets"


if __name__ == "__main__":
	if not torch.cuda.is_available():
		device = "cpu"
	elif device is None or device < 0:
		import random
		if torch.cuda.device_count() == 1:
			device = "cuda"
		else:
			device = f"cuda:{random.randint(0, torch.cuda.device_count() - 1)}"
	device = torch.device(device)
	if mode == "decode":
		# Read using a parallel thread; this avoids delays from blocking
		import concurrent.futures
		exc = concurrent.futures.ThreadPoolExecutor(max_workers=1)
		it = stream_from_file(file, device=device)
		try:
			limit = 1
			rescale = False
			wav, sr = next(it)
			fut = None
			while True:
				if fut:
					wav, sr = fut.result()
				fut = exc.submit(next, it)
				# Rescale shouldn't be needed so this'll just be skipped
				if rescale:
					mx = wav.abs().max()
					wav = wav * min(limit / mx, 1)
				else:
					wav = wav.clamp(-limit, limit)
				# Must convert to appropriate s16le c2 format
				wav *= 32767
				wav = wav.T.to(torch.int16)
				# sys.stderr.write(f"{wav.shape}\n")
				length, channels = wav.shape
				if channels != 2:
					wav = wav.expand(length, 2)
				# Pytorch does not allow direct serialisation & .data does not work because it's not contiguous
				b = wav.cpu().numpy().tobytes()
				sys.stdout.buffer.write(b)
		except StopIteration:
			pass
		exc.shutdown()
	else:
		stream_to_file(file, name=name, bitrate=bitrate, device=device)