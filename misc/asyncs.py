import asyncio
import collections
import concurrent.futures
import contextlib
import functools
import inspect
import random
import threading
import time
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from time import time as utc
from traceback import print_exc

print("ASYNCS:", __name__)

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
	fut = asyncio.Future()
	cst(fut.set_result, obj)
	return fut


# Thread pool manager for multithreaded operations.
class MultiThreadPool(collections.abc.Sized, concurrent.futures.Executor):

	def __init__(self, thread_count=8, pool_count=1, initializer=None, executor=ThreadPoolExecutor):
		self.pools = []
		self.pool_count = max(1, pool_count)
		self.thread_count = max(1, thread_count)
		self.initializer = initializer
		self.executor = executor
		self.position = -1
		self.update()
		self.tasks = weakref.WeakSet()

	def __len__(self): sum(len(pool._threads) for pool in self.pools)

	# Adjusts pool count if necessary
	def _update(self):
		if self.pool_count != len(self.pools):
			self.pool_count = max(1, self.pool_count)
			self.thread_count = max(1, self.thread_count)
			while self.pool_count > len(self.pools):
				pool = self.executor(
					max_workers=self.thread_count,
					initializer=self.initializer,
				)
				self.pools.append(pool)
			while self.pool_count < len(self.pools):
				func = self.pools.popright().shutdown
				self.pools[-1].submit(func, wait=True)

	def update(self):
		if not self.pools:
			return self._update()
		self.position = (self.position + 1) % len(self.pools)
		pool = random.choice(self.pools)
		if getattr(pool, "_shutdown", False) or getattr(pool, "_broken", False):
			return
		pool.submit(self._update)

	def map(self, func, *args, **kwargs):
		self.update()
		fut = self.pools[self.position].map(func, *args, **kwargs)
		self.tasks.add(fut)
		return fut

	def submit(self, func, *args, **kwargs):
		self.update()
		fut = self.pools[self.position].submit(func, *args, **kwargs)
		self.tasks.add(fut)
		return fut

	def shutdown(self, wait=True):
		return [exc.shutdown(wait) for exc in self.pools].append(self.pools.clear())

mthreads = ThreadPoolExecutor(7, initializer=__setloop__)
lim = 8
# bthreads = MultiThreadPool(lim, initializer=__setloop__)
# athreads = concurrent.futures.exc_worker = MultiThreadPool(lim * 2, pool_count=2, initializer=__setloop__)
bthreads = ThreadPoolExecutor(max_workers=lim, initializer=__setloop__)
athreads = ThreadPoolExecutor(max_workers=lim * 4, initializer=__setloop__)

def get_executor(priority):
	if priority not in range(0, 2):
		return
	return (athreads, bthreads)[priority]

def initialise_ppe():
	global athreads, bthreads, pthreads, get_executor
	athreads.shutdown(wait=False)
	bthreads.shutdown(wait=False)
	lim = 32
	# bthreads = MultiThreadPool(lim, initializer=__setloop__)
	# athreads = concurrent.futures.exc_worker = MultiThreadPool(lim * 2, pool_count=2, initializer=__setloop__)
	bthreads = ThreadPoolExecutor(max_workers=lim, initializer=__setloop__)
	athreads = ThreadPoolExecutor(max_workers=lim * 4, initializer=__setloop__)

	from concurrent.futures import ProcessPoolExecutor
	pthreads = ProcessPoolExecutor(max_workers=2)
	# pthreads = MultiThreadPool(6, executor=ProcessPoolExecutor)

	def get_executor(priority):
		if priority not in range(0, 4):
			return
		return (athreads, bthreads, pthreads, mthreads)[priority]


# Checks if an object can be used in "await" operations.
def awaitable(obj) -> bool:
	if isinstance(obj, type):
		return False
	return hasattr(obj, "__await__") or isinstance(obj, asyncio.Future) or inspect.isawaitable(obj)

# Async function that waits for a given time interval if the result of the input coroutine is None.
async def wait_on_none(coro, seconds=0.5):
	resp = await coro
	if resp is None:
		await asyncio.sleep(seconds)
	return resp

def get_event_loop():
	return eloop

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

def create_thread(func, *args, **kwargs) -> threading.Thread:
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
	if not isinstance(obj, asyncio.Future):
		obj = csubmit(obj, loop=loop)#, name=f"Task-{utc()}-{obj}-{lim_str(str((args, kwargs)), 256)}")
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
	"Blocking call that waits for a single asyncio future to complete, do *not* call from main asyncio loop."
	return convert_fut(fut).result(timeout=timeout)

def convert_fut(fut) -> Future:
	loop = get_event_loop()
	if is_main_thread():
		raise RuntimeError("This function must not be called from the main thread's asyncio loop.")
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
			yield await_fut(anext(ait))
	except StopAsyncIteration:
		pass

def is_main_thread() -> bool:
	return threading.current_thread() is threading.main_thread()

# A dummy coroutine that returns None.
async def async_nop(*args, **kwargs):
	pass

async def delayed_coro(fut, duration=None):
	async with Delay(duration):
		return await fut

async def waited_coro(fut, duration=None):
	await asyncio.sleep(duration)
	return await fut

async def traceback_coro(fut, *args):
	try:
		return await fut
	except args:
		pass
	except Exception:
		print_exc()

def trace(fut, *args):
	return csubmit(traceback_coro(fut, *args))

def throw(exc):
	raise exc


class EmptyContext(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):
	"An empty context manager that has no effect. Serves as compatibility with dynamically replaceable contexts."
	def __enter__(self, *args):
		return self
	def __exit__(*args):
		pass
	def __aenter__(self, *args):
		return as_fut(self)
	def __aexit__(*args):
		return emptyfut
	def __call__(self, *args):
		return self
	busy = False
	active = False
emptyctx = EmptyContext()

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
	"Manages concurrency limits, similar to asyncio.Semaphore, but has a secondary threshold for enqueued tasks, as well as an optional rate limiter. Compatible with both sync and async contexts."

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
				self.mutex.release()
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
	"A context manager that delays the return of a function call."

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