import asyncio
import collections
import concurrent.futures
import contextlib
import functools
import inspect
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, thread
from time import time as utc
from misc.types import lim_str

print("ASYNCS:", __name__)

def is_main_thread() -> bool:
	return threading.current_thread() is threading.main_thread()
def get_event_loop():
	try:
		return asyncio.get_event_loop()
	except RuntimeError:
		return eloop

# Main event loop for all asyncio operations.
try:
	eloop = asyncio.get_event_loop()
except Exception:
	eloop = asyncio.new_event_loop()
def __setloop__(): return asyncio.set_event_loop(eloop)
__setloop__()

def at():
	return asyncio.all_tasks(eloop)

emptyfut = fut_nop = asyncio.Future(loop=eloop)
fut_nop.set_result(None)
Future = concurrent.futures.Future
newfut = nullfut = Future()
newfut.set_result(None)

collections.__dict__.update(collections.abc.__dict__)

def as_fut(obj):
	if obj is None:
		return emptyfut
	fut = asyncio.Future(loop=eloop)
	cst(fut.set_result, obj)
	return fut

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
			daemon=True,
		)
		t.start()
		self._threads.add(t)
		thread._threads_queues[t] = self._work_queue

concurrent.futures.ThreadPoolExecutor._adjust_thread_count = _adjust_thread_count

mthreads = ThreadPoolExecutor(7, initializer=__setloop__)
lim = 8
bthreads = ThreadPoolExecutor(max_workers=lim, initializer=__setloop__)
athreads = ThreadPoolExecutor(max_workers=lim * 4, initializer=__setloop__)

def get_executor(priority):
	if priority not in range(0, 2):
		raise NotImplementedError(priority)
	return (athreads, bthreads)[priority]

def initialise_ppe():
	global athreads, bthreads, pthreads, get_executor
	athreads.shutdown(wait=False)
	bthreads.shutdown(wait=False)
	lim = 32
	bthreads = ThreadPoolExecutor(max_workers=lim, initializer=__setloop__)
	athreads = ThreadPoolExecutor(max_workers=lim * 4, initializer=__setloop__)

	from concurrent.futures import ProcessPoolExecutor
	pthreads = ProcessPoolExecutor(max_workers=2)

	def get_executor(priority):
		if priority not in range(0, 4):
			raise NotImplementedError(priority)
		return (athreads, bthreads, pthreads, mthreads)[priority]


# Checks if an object can be used in "await" operations.
def awaitable(obj) -> bool:
	if isinstance(obj, type):
		return False
	return callable(getattr(obj, "__await__", None)) or isinstance(obj, asyncio.Future) or inspect.isawaitable(obj)

# Async function that waits for a given time interval if the result of the input coroutine is None.
async def wait_on_none(coro, seconds=0.5):
	resp = await coro
	if resp is None:
		await asyncio.sleep(seconds)
	return resp

def wrap_future(fut, loop=None, shield=False, thread_safe=True) -> asyncio.Future:
	"Creates an asyncio Future that waits on a multithreaded one."
	if getattr(fut, "done", None) and fut.done():
		res = fut.result()
		return as_fut(res)
	if loop is None:
		loop = get_event_loop()
	wrapper = None
	if not thread_safe:
		try:
			wrapper = asyncio.wrap_future(fut, loop=loop)
		except (AttributeError, TypeError):
			pass
	if wrapper is None:
		wrapper = loop.create_future()

		def set_suppress(res, is_exception=False):
			try:
				if is_exception:
					if isinstance(res, StopIteration):
						res = RuntimeError(res)
					wrapper.set_exception(res)
				else:
					wrapper.set_result(res)
			except (RuntimeError, asyncio.InvalidStateError):
				pass

		def on_done(*void):
			if loop.is_closed():
				return
			try:
				res = fut.result()
			except Exception as ex:
				loop.call_soon_threadsafe(set_suppress, ex, True)
			else:
				loop.call_soon_threadsafe(set_suppress, res)

		fut.add_done_callback(on_done)
	if shield:
		wrapper = asyncio.shield(wrapper)
	return wrapper

def shutdown_thread_after(thread, fut):
	fut.result()
	return thread.shutdown(wait=True)

def create_thread(func, *args, **kwargs) -> Future:
	fut = Future()
	def target():
		try:
			resp = func(*args, **kwargs)
		except BaseException as ex:
			if isinstance(ex, StopIteration):
				ex = RuntimeError(ex)
			fut.set_exception(ex)
			raise
		else:
			fut.set_result(resp)
		return resp
	t = threading.Thread(
		target=target,
		daemon=True,
	)
	t.start()
	return fut
tsubmit = create_thread

def create_future_ex(func, *args, timeout=None, priority=False, **kwargs) -> Future:
	"Runs a function call in a parallel thread, returning a future object waiting on the output."
	try:
		kwargs["timeout"] = kwargs.pop("_timeout_")
	except KeyError:
		pass
	executor = get_executor(priority)
	if executor is None:
		# print(f"Creating thread {priority}: {func} {args} {kwargs}")
		return tsubmit(func, *args, **kwargs)
	fut = executor.submit(func, *args, **kwargs)
	if timeout is not None:
		fut = executor.submit(fut.result, timeout=timeout)
	return fut
esubmit = create_future_ex

def create_future(obj, *args, loop=None, timeout=None, priority=False, thread_safe=True, **kwargs) -> asyncio.Future:
	"High level future asyncio creation function that accepts both sync and async functions, as well as coroutines directly."
	if loop is None:
		loop = get_event_loop()
	if loop.is_closed():
		return emptyfut
	if asyncio.iscoroutinefunction(obj):
		obj = obj(*args, **kwargs)
	if callable(obj):
		if asyncio.iscoroutinefunction(obj.__call__) or priority is None:
			obj = obj.__call__(*args, **kwargs)
		else:
			executor = get_executor(priority)
			if executor is not None:
				if kwargs:
					try:
						kwargs["timeout"] = kwargs.pop("_timeout_")
					except KeyError:
						pass
					obj = functools.partial(obj, *args, **kwargs)
					args = ()
				obj = loop.run_in_executor(executor, obj, *args)
			else:
				obj = wrap_future(esubmit(obj, *args, timeout=timeout, priority=priority, **kwargs), loop=loop, thread_safe=thread_safe)
	if obj is None:
		return emptyfut
	if not isinstance(obj, asyncio.Future):
		try:
			return loop.create_task(obj)
		except RuntimeError:
			return create_task(obj, name=f"Task-{utc()}-{obj}-{lim_str(str((args, kwargs)), 256)}")
	return obj
asubmit = create_future

def create_task(fut, *args, loop=None, **kwargs) -> asyncio.Future:
	"Creates an asyncio Task object from an awaitable object."
	if loop is None:
		loop = get_event_loop()
	if fut is None or loop.is_closed():
		return emptyfut
	if asyncio.iscoroutinefunction(fut):
		fut = fut(*args, **kwargs)
	if not is_main_thread() and asyncio.iscoroutine(fut):
		return wrap_future(asyncio.run_coroutine_threadsafe(fut, loop=loop))
	if asyncio.iscoroutine(fut):
		return loop.create_task(fut, **kwargs)
	try:
		return asyncio.ensure_future(fut, loop=loop)
	except TypeError:
		print(type(fut), fut)
		raise
fsubmit = csubmit = create_task

def cst(fut, *args, **kwargs):
	loop = get_event_loop()
	if loop.is_closed():
		return
	if not loop.is_running():
		return fut(*args, **kwargs)
	return loop.call_soon_threadsafe(fut, *args, **kwargs)

def pipe_fut(src, dest):
	try:
		res = src.result()
		dest.set_result(res)
	except BaseException as e:
		if isinstance(e, StopIteration):
			e = RuntimeError(e)
		dest.set_exception(e)
	return res

async def gather(*futs, return_exceptions=False):
	"Same functionality as asyncio.gather, but with fallback to main thread if called from a non-asyncio loop."
	if not is_main_thread() and any(map(asyncio.iscoroutine, futs)):
		futs = [asyncio.run_coroutine_threadsafe(fut, loop=get_event_loop()) for fut in futs]
		out = []
		if return_exceptions:
			for fut in futs:
				try:
					res = await wrap_future(fut)
				except Exception as ex:
					res = ex
				out.append(res)
		else:
			for fut in futs:
				res = await wrap_future(fut)
				out.append(res)
		return out
	return await asyncio.gather(*futs, return_exceptions=return_exceptions)

async def _await_fut(fut, ret) -> Future:
	try:
		out = await fut
	except BaseException as ex:
		if isinstance(ex, StopIteration):
			ex = RuntimeError(ex)
		ret.set_exception(ex)
	else:
		ret.set_result(out)
	return ret

def await_fut(fut, timeout=None):
	"Blocking call that waits for a single asyncio future to complete."
	if is_main_thread():
		if timeout:
			fut = asyncio.wait_for(fut, timeout=timeout)
		return asyncio.run(fut)
	return convert_fut(fut).result(timeout=timeout)

def convert_fut(fut):
	if is_main_thread():
		fut = concurrent.futures.Future()
		fut.set_result(asyncio.run(fut))
		return fut
	loop = get_event_loop()
	try:
		ret = asyncio.run_coroutine_threadsafe(fut, loop=loop)
	except Exception:
		ret = Future()
		asyncio.run_coroutine_threadsafe(_await_fut(fut, ret), loop=loop)
	return ret

async def flatten(ait) -> list:
	resp = []
	async for x in ait:
		resp.append(x)
	return resp

async def unflatten(it):
	try:
		while True:
			yield await asubmit(next, it)
	except StopIteration:
		pass

def reflatten(ait):
	try:
		while True:
			yield await_fut(anext(ait))  # noqa: F821
	except StopAsyncIteration:
		pass

# A dummy coroutine that returns None.
async def async_nop(*args, **kwargs):
	pass

async def delayed_coro(fut, duration=None):
	async with Delay(duration):
		return await fut

def delayed_sync(fut, duration=None):
	with Delay(duration):
		return fut.result()

async def waited_coro(fut, duration=None):
	await asyncio.sleep(duration)
	return await fut

def waited_sync(fut, duration=None):
	time.sleep(duration)
	return fut.result()

async def delayed_callback(fut, delay, func, *args, repeat=False, exc=False, **kwargs):
	"A function that takes a coroutine/task, and calls a second function if it takes longer than the specified delay."
	await asyncio.sleep(delay / 2)
	if not fut.done():
		await asyncio.sleep(delay / 2)
	try:
		return fut.result(), False
	except Exception as ex:
		if exc and not isinstance(ex, asyncio.exceptions.InvalidStateError):
			raise
		while not fut.done():
			async with Delay(repeat):
				if hasattr(func, "__call__"):
					res = func(*args, **kwargs)
				else:
					res = func
				if awaitable(res):
					await res
			if not repeat:
				break
		return await fut, True

async def traceback_coro(fut, *args):
	try:
		return await fut
	except args:
		pass
	except Exception:
		traceback.print_exc()

def format_async_stack(coro, limit=64):
	"""Walk coroutine/awaitable chain and return formatted stack trace."""
	frames = []
	while coro and limit > 0:
		if inspect.iscoroutine(coro):
			frame = coro.cr_frame
			if frame:
				frames.extend(traceback.format_stack(frame))
			coro = coro.cr_await
		elif inspect.isgenerator(coro):
			frame = coro.gi_frame
			if frame:
				frames.extend(traceback.format_stack(frame))
			coro = coro.gi_yieldfrom
		else:
			break
		limit -= 1
	return frames

def trace(fut, *args):
	return csubmit(traceback_coro(fut, *args))

def throw(exc):
	raise exc

emptyctx = contextlib.nullcontext()

class CloseableAsyncIterator:

	def __init__(self, it, close=None):
		self.it = it
		self.fclose = close

	def __next__(self):
		return self.it.__next__()

	def __iter__(self):
		return iter(self.it)

	async def __anext__(self):
		return await self.it.__anext__()

	async def __aiter__(self):
		async for item in self.it:
			yield item

	async def __enter__(self):
		return self

	async def __exit__(self):
		out = self.fclose
		out = out() if callable(out) else out
		if awaitable(out):
			return await out
		return out

	def close(self):
		out = self.fclose
		out = out() if callable(out) else out
		out = csubmit(out) if awaitable(out) else as_fut(out)
		return out

	def __del__(self):
		self.close()

class Mutex:
	"""A thread-safe and async-compatible mutex lock implementation.
	This class provides both synchronous and asynchronous locking mechanisms using a combination
	of threading.Lock and asyncio.Condition.
	Attributes:
		lock (threading.Lock): The underlying threading lock object
		locked (bool): Current state of the mutex
		cond (threading.Condition): Condition variable for synchronous operations
		acond (asyncio.Condition): Condition variable for asynchronous operations
	Methods:
		signal(): Notifies waiting threads/coroutines that the lock has been released
		acquire_sync(): Synchronously acquires the mutex lock
		release_sync(): Synchronously releases the mutex lock
		acquire_async(): Asynchronously acquires the mutex lock
		release_async(): Asynchronously releases the mutex lock
	Example:
		# Synchronous usage
		mutex = Mutex()
		mutex.acquire_sync()
			# Critical section
			pass
			mutex.release_sync()
		# Asynchronous usage
		mutex = Mutex()
		await mutex.acquire_async()
			# Critical section
			pass
			await mutex.release_async()
	"""

	def __init__(self):
		self.lock = threading.Lock()
		self.locked = False
		self.cond = threading.Condition()
		self.acond = asyncio.Condition()

	def __str__(self):
		return self.__class__.__name__ + " " + ("[Locked]" if self.locked else "[Unlocked]")

	def signal(self):
		with self.cond:
			self.cond.notify()
		loop = get_event_loop()
		if loop and loop.is_running():
			async def _signal(self):
				async with self.acond:
					self.acond.notify()
			asyncio.run_coroutine_threadsafe(_signal(self), loop=loop)
		return self

	def acquire_sync(self):
		with self.lock:
			while self.locked:
				self.lock.release()
				try:
					with self.cond:
						self.cond.wait()
				finally:
					self.lock.acquire()
			self.locked = True
		return self

	def release_sync(self):
		self.locked = False
		return self.signal()

	async def acquire_async(self):
		with self.lock:
			while self.locked:
				self.lock.release()
				try:
					async with self.acond:
						await self.acond.wait()
				finally:
					self.lock.acquire()
			self.locked = True
		return self

	async def release_async(self):
		self.locked = False
		return self.signal()

class Semaphore(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):
	"""A flexible synchronization primitive that combines features of threading and asyncio semaphores.
	This Semaphore implementation provides both synchronous and asynchronous context manager interfaces,
	with additional rate limiting and LIFO/FIFO queueing capabilities.
	Args:
		limit (int, optional): Maximum number of concurrent active operations. Defaults to 256.
		buffer (int, optional): Maximum number of waiting operations. Defaults to 32.
		rate_limit (float, optional): Time in seconds between allowed operations. Defaults to None.
		sync (bool, optional): Whether to synchronize with rate_limit boundaries. Defaults to False.
		lifo (bool, optional): Whether to use LIFO instead of FIFO ordering. Defaults to False.
	Attributes:
		active (int): Current number of active operations
		passive (int): Current number of waiting operations
		paused (bool): Whether the semaphore is currently paused
		rate_bin (collections.deque): Timestamps of recent operations for rate limiting
		reset_after (float): Time until rate limit resets
	Example:
		```
		# Synchronous usage
		with Semaphore(limit=5) as sem:
			# Protected code here
		# Asynchronous usage
		async with Semaphore(limit=5) as sem:
			# Protected async code here
		# Rate limited usage
		sem = Semaphore(limit=100, rate_limit=1.0)  # 100 operations per second
		```
	Raises:
		SemaphoreOverflowError: When buffer limit is exceeded
		SystemError: On unexpected underflow of active or passive counters
	"""

	__slots__ = ("limit", "buffer", "active", "passive", "rate_limit", "rate_bin", "lifo", "traces", "tempfut")
	TRACE = 0

	def __init__(self, limit=256, buffer=32, rate_limit=None, sync=False, lifo=False):
		self.limit = limit
		self.buffer = buffer
		self.active = 0
		self.passive = 0
		self.rate_limit = rate_limit
		self.rate_bin = deque()
		self.mutex = threading.Lock()
		self.cond = threading.Condition()
		self.acond = asyncio.Condition()
		self.lifo = lifo
		self.sync = sync
		self.paused = False
		self.tempfut = None
		if self.TRACE:
			self.traces = {}

	def __str__(self):
		classname = str(self.__class__).replace("'>", "")
		classname = classname[classname.index("'") + 1:]
		s = f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>: {self.active}/{self.limit}, {self.passive}/{self.buffer}"
		if self.rate_limit:
			s += f", {round(self.reset_after, 1)}/{self.rate_limit}"
		if self.paused:
			s += " [PAUSED]"
		return s

	@property
	def reset_after(self):
		if not self.rate_limit or not self.rate_bin:
			return 0
		t = time.time()
		if t - self.rate_bin[0] <= self.rate_limit:
			return self.rate_limit - (t - self.rate_bin[0])
		return 0

	def update_bin(self):
		if self.rate_limit:
			# was_full = len(self.rate_bin) >= self.limit
			try:
				if self.lifo:
					if self.rate_bin and time.time() - self.rate_bin[-1] >= self.rate_limit:
						self.rate_bin.clear()
						self.signal()
				else:
					while self.rate_bin and time.time() - self.rate_bin[0] >= self.rate_limit:
						self.rate_bin.popleft()
						self.signal()
			except IndexError:
				pass
			# if was_full and len(self.rate_bin) < self.limit:
			# 	self.signal()
		return self.rate_bin

	def delay_for(self, seconds=0):
		self.mutex.acquire(timeout=1)
		self.paused = True
		def undelay():
			time.sleep(seconds)
			self.paused = False
			self.mutex.release()
		func = esubmit if seconds < 60 else tsubmit
		return func(undelay)

	def enter(self):
		self.active += 1
		if self.rate_limit:
			t = time.time()
			self.update_bin().append(t)
		return self

	def check_overflow(self):
		if self.is_full():
			raise SemaphoreOverflowError(f"Semaphore object of limit {self.limit} overloaded by {self.passive}")

	def signal(self):
		if not self.passive:
			return self
		with self.cond:
			self.cond.notify_all()
		loop = get_event_loop()
		if loop and loop.is_running():
			async def _signal(self):
				async with self.acond:
					self.acond.notify_all()
			asyncio.run_coroutine_threadsafe(_signal(self), loop=loop)
		return self

	def __enter__(self):
		self.check_overflow()
		self.mutex.acquire(timeout=1)
		self.passive += 1
		if self.TRACE:
			ts = time.time_ns()
			while ts in self.traces:
				ts += 1
			self.traces[ts] = inspect.stack()
		try:
			while self.is_busy():
				try:
					self.mutex.release()
				except RuntimeError:
					pass
				try:
					if self.paused:
						time.sleep(1)
						continue
					if self.sync:
						rem = (time.time() / self.rate_limit) % 1
						if rem > 1 / 60:
							time.sleep((1 - rem) * self.rate_limit)
							continue
					if self.rate_bin:
						remaining = self.rate_limit - time.time() + self.rate_bin[0 - self.lifo]
						if remaining > 0:
							time.sleep(remaining)
							continue
						if len(self.rate_bin) >= self.limit:
							time.sleep(0.005)
							continue
					with self.cond:
						self.cond.wait()
				finally:
					self.mutex.acquire(timeout=1)
			return self.enter()
		finally:
			if self.TRACE:
				self.traces.pop(ts)
			self.passive -= 1
			self.mutex.release()
			if self.passive < 0:
				raise SystemError("Unexpected Semaphore passive underflow!")

	def __exit__(self, *args):
		self.mutex.acquire(timeout=1)
		try:
			self.active -= 1
			if self.active < 0:
				raise SystemError("Unexpected Semaphore active underflow!")
			self.update_bin()
			self.signal()
		finally:
			self.mutex.release()

	async def __aenter__(self):
		self.check_overflow()
		self.mutex.acquire(timeout=1)
		self.passive += 1
		if self.TRACE:
			ts = time.time_ns()
			while ts in self.traces:
				ts += 1
			self.traces[ts] = inspect.stack()
		try:
			while self.is_busy():
				self.mutex.release()
				try:
					if self.paused:
						await asyncio.sleep(1)
						continue
					if self.sync:
						rem = (time.time() / self.rate_limit) % 1
						if rem > 1 / 60:
							await asyncio.sleep((1 - rem) * self.rate_limit)
							continue
					if self.rate_bin:
						remaining = self.rate_limit - time.time() + self.rate_bin[0 - self.lifo]
						if remaining > 0:
							await asyncio.sleep(remaining)
							continue
						if len(self.rate_bin) >= self.limit:
							await asyncio.sleep(0.005)
							continue
					async with self.acond:
						await self.acond.wait()
				finally:
					self.mutex.acquire(timeout=1)
			return self.enter()
		finally:
			if self.TRACE:
				self.traces.pop(ts)
			self.passive -= 1
			self.mutex.release()
			if self.passive < 0:
				raise SystemError("Unexpected Semaphore passive underflow!")

	async def __aexit__(self, *args):
		self.mutex.acquire(timeout=1)
		try:
			self.active -= 1
			if self.active < 0:
				raise SystemError("Unexpected Semaphore active underflow!")
			self.update_bin()
			self.signal()
		finally:
			self.mutex.release()

	def wait(self):
		self.__enter__()
		self.__exit__()

	async def __call__(self):
		await self.__aenter__()
		await self.__aexit__()
	
	acquire = __call__

	def pause(self):
		with self:
			self.paused = True
		return self

	async def apause(self):
		async with self:
			self.paused = True
		return self

	def unpause(self):
		self.paused = False
		return self.signal()

	resume = unpause

	def finish(self):
		with self.cond:
			self.cond.wait_for(self.is_not_used)

	async def afinish(self):
		async with self.acond:
			await self.acond.wait_for(self.is_not_used)

	def is_active(self):
		return self.active or self.passive

	def is_used(self):
		return self.active or self.is_busy()

	def is_not_used(self):
		return not self.active and self.is_free()

	def is_busy(self):
		if self.rate_limit and len(self.update_bin()) >= self.limit:
			return True
		return self.paused or self.active >= self.limit

	def is_free(self):
		return not self.is_busy()

	def is_full(self):
		return self.passive >= self.buffer and self.is_busy()

	def clear(self):
		self.rate_bin.clear()

	@property
	def full(self):
		return self.is_full()

	@property
	def busy(self):
		return self.is_busy()

	@property
	def free(self):
		return self.is_free()

class SemaphoreOverflowError(RuntimeError):
	__slots__ = ("__weakref__",)


class Delay(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):
	"""A context manager and decorator for creating delays in both synchronous and asynchronous code.
	This class implements a delay mechanism that can be used as a context manager,
	decorator, or callable in both synchronous and asynchronous contexts. It ensures
	a minimum duration passes between entering and exiting the context.
	Args:
		duration (float, optional): The minimum time in seconds that should elapse. Defaults to 0.
	Examples:
		As a context manager:
			>>> with Delay(1):
			...     do_something()  # This block will take at least 1 second
		As an async context manager:
			>>> async with Delay(1):
			...     await do_something()  # This block will take at least 1 second
		As a decorator:
			>>> @Delay(1)
			... def function():
			...     pass  # This function will take at least 1 second
		As an async decorator:
			>>> @Delay(1)
			... async def async_function():
			...     pass  # This function will take at least 1 second
		As a callable:
			>>> delay = Delay(1)
			>>> do_something()
			>>> delay()  # Will sleep for remaining time if less than 1 second has passed
	"""

	def __init__(self, duration=0):
		self.duration = duration
		self.start = utc()

	def __call__(self):
		return self.exit()

	def __exit__(self, *args):
		remaining = self.duration - utc() + self.start
		if remaining > 0:
			time.sleep(remaining)

	async def __aexit__(self, *args):
		remaining = self.duration - utc() + self.start
		if remaining > 0:
			await asyncio.sleep(remaining)


class AsyncStreamWrapper:
	def __init__(self, async_iterator):
		self.async_iterator = async_iterator
		self.buffer = bytearray()
		self._closed = False

	async def read(self, n=-1):
		if self._closed:
			raise ValueError("I/O operation on closed file.")
		
		async for chunk in self.async_iterator:
			self.buffer.extend(chunk)
			if n >= 0 and len(self.buffer) >= n:
				break

		if n < 0:  # Read all remaining data
			data = bytes(self.buffer)
			self.buffer.clear()
		else:
			data = bytes(self.buffer[:n])
			self.buffer = self.buffer[n:]
		
		return data

	def close(self):
		self._closed = True

	async def __aenter__(self):
		return self

	async def __aexit__(self, exc_type, exc_value, traceback):
		self.close()