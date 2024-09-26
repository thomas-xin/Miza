import bisect
from collections import deque
from math import ceil, floor
import pickle
import random
import threading
import unittest
# import numpy as np
from ring_vector import RingVector
CustomArray = CustomList = CustomCircularBuffer = CircularBuffer = ThreadSafeList = RingVector


class TestCustomArray(unittest.TestCase):
	def setUp(self):
		self.array = CustomArray()

	def test_append(self):
		self.array.append(1)
		self.assertEqual(self.array[0], 1)
		self.assertEqual(len(self.array), 1)

	def test_appendleft(self):
		self.array.appendleft(1)
		self.assertEqual(self.array[0], 1)
		self.assertEqual(len(self.array), 1)

	def test_extend(self):
		self.array.extend([1, 2, 3])
		self.assertEqual(self.array[0], 1)
		self.assertEqual(self.array[1], 2)
		self.assertEqual(self.array[2], 3)
		self.assertEqual(len(self.array), 3)

	def test_extendleft(self):
		self.array.extendleft([1, 2, 3])
		self.assertEqual(self.array[0], 3)
		self.assertEqual(self.array[1], 2)
		self.assertEqual(self.array[2], 1)
		self.assertEqual(len(self.array), 3)

	def test_pop(self):
		self.array.append(1)
		value = self.array.pop()
		self.assertEqual(value, 1)
		self.assertEqual(len(self.array), 0)

	def test_popleft(self):
		self.array.append(1)
		value = self.array.popleft()
		self.assertEqual(value, 1)
		self.assertEqual(len(self.array), 0)

	def test_insert(self):
		self.array.append(1)
		self.array.insert(0, 2)
		self.assertEqual(self.array[0], 2)
		self.assertEqual(self.array[1], 1)

	def test_remove(self):
		self.array.append(1)
		self.array.append(2)
		self.array.remove(1)
		self.assertEqual(self.array[0], 2)
		self.assertEqual(len(self.array), 1)

	def test_index(self):
		self.array.append(1)
		self.array.append(2)
		self.assertEqual(self.array.index(2), 1)

	def test_count(self):
		self.array.append(1)
		self.array.append(1)
		self.assertEqual(self.array.count(1), 2)

	def test_sort(self):
		self.array.extend([3, 1, 2])
		self.array.sort()
		self.assertEqual(self.array[0], 1)
		self.assertEqual(self.array[1], 2)
		self.assertEqual(self.array[2], 3)

	def test_reverse(self):
		self.array.extend([1, 2, 3])
		self.array.reverse()
		self.assertEqual(self.array[0], 3)
		self.assertEqual(self.array[1], 2)
		self.assertEqual(self.array[2], 1)


class TestCustomList(unittest.TestCase):
	def test_initialization(self):
		custom_list = CustomList(1, 2, 3)
		self.assertEqual(custom_list, [1, 2, 3])
	
	def test_append(self):
		custom_list = CustomList()
		custom_list.append(4)
		self.assertEqual(custom_list, [4])
	
	def test_remove_existing_item(self):
		custom_list = CustomList(1, 2, 3)
		custom_list.remove(2)
		self.assertEqual(custom_list, [1, 3])
	
	def test_remove_non_existing_item(self):
		custom_list = CustomList(1, 2, 3)
		with self.assertRaises(ValueError):
			custom_list.remove(4)
	
	def test_get_item(self):
		custom_list = CustomList(1, 2, 3)
		self.assertEqual(custom_list[1], 2)
	
	def test_length(self):
		custom_list = CustomList(1, 2, 3)
		self.assertEqual(len(custom_list), 3)

	def test_split(self):
		custom_list = CustomList(1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9)
		new_list = custom_list.split(0)
		self.assertEqual(new_list, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

	def test_join(self):
		custom_list = CustomList(1, 2, 3)
		new_list = custom_list.join([4, 5, 6])
		self.assertEqual(new_list, [4, 1, 2, 3, 5, 1, 2, 3, 6])

	def test_delitems(self):
		custom_list = CustomList(1, 2, 3, 4, 5, 6, 7, 8, 9)
		custom_list.delitems([0, 3, 2, 6])
		self.assertEqual(custom_list, [2, 5, 6, 8, 9])


class TestCustomCircularBuffer(unittest.TestCase):
	def setUp(self):
		self.capacity = 5
		self.buffer = CustomCircularBuffer()

	def test_enqueue_dequeue(self):
		# Test simple enqueue and dequeue
		self.buffer.enqueue(1)
		self.buffer.enqueue(2)
		self.buffer.enqueue(3)
		self.assertEqual(self.buffer.dequeue(), 1)
		self.assertEqual(self.buffer.dequeue(), 2)
		self.assertEqual(self.buffer.dequeue(), 3)
	
	def test_full_buffer(self):
		# Test buffer full condition
		for i in range(self.capacity):
			self.buffer.enqueue(i)
		self.assertEqual(len(self.buffer), self.capacity)
		# with self.assertRaises(IndexError):
		# 	self.buffer.enqueue(10)

	def test_empty_buffer(self):
		# Test buffer empty condition
		with self.assertRaises(IndexError):
			self.buffer.dequeue()

	def test_wraparound_enqueue_dequeue(self):
		# Test wraparound logic
		for i in range(self.capacity):
			self.buffer.enqueue(i)
		self.assertEqual(self.buffer.dequeue(), 0)
		self.buffer.enqueue(5)
		self.assertEqual(self.buffer.dequeue(), 1)
		self.assertEqual(self.buffer.dequeue(), 2)
		self.assertEqual(self.buffer.dequeue(), 3)
		self.assertEqual(self.buffer.dequeue(), 4)
		self.assertEqual(self.buffer.dequeue(), 5)

	def test_random_operations(self):
		# Test random operations extensively
		operations = 10000
		reference = []
		
		for _ in range(operations):
			operation = random.choice(["enqueue", "dequeue"])
			if operation == "enqueue":
				if len(reference) < self.capacity:
					value = random.randint(0, 1000)
					reference.append(value)
					self.buffer.enqueue(value)
				else:
					self.assertEqual(len(self.buffer), self.capacity)
					# with self.assertRaises(IndexError):
					# 	self.buffer.enqueue(random.randint(0, 1000))
			elif operation == "dequeue":
				if reference:
					self.assertEqual(self.buffer.dequeue(), reference.pop(0))
				else:
					with self.assertRaises(IndexError):
						self.buffer.dequeue()
		
		# Check final state
		while reference:
			self.assertEqual(self.buffer.dequeue(), reference.pop(0))

	def test_len(self):
		# Test the length function
		self.assertEqual(len(self.buffer), 0)
		self.buffer.enqueue(1)
		self.assertEqual(len(self.buffer), 1)
		self.buffer.enqueue(2)
		self.assertEqual(len(self.buffer), 2)
		self.buffer.dequeue()
		self.assertEqual(len(self.buffer), 1)
		self.buffer.dequeue()
		self.assertEqual(len(self.buffer), 0)


class TestCircularBuffer(unittest.TestCase):
	
	def setUp(self):
		self.buffer_size = 10
		self.circular_buffer = CircularBuffer()
	
	def test_append(self):
		for i in range(self.buffer_size):
			self.circular_buffer.append(i)
		self.assertEqual(self.circular_buffer, range(self.buffer_size))
		
		# Test appending past full capacity
		self.circular_buffer.append(10)
		self.assertEqual(self.circular_buffer, range(0, 11))

	def test_insert(self):
		for i in range(5):
			self.circular_buffer.append(i)
		
		self.circular_buffer.insert(2, 99)
		self.assertEqual(self.circular_buffer, [0, 1, 99, 2, 3, 4])
		
		# Test inserting past the end
		self.circular_buffer.insert(10, 100)
		self.assertEqual(self.circular_buffer, [0, 1, 99, 2, 3, 4, 100])

	def test_remove(self):
		# Test removing element from empty buffer
		with self.assertRaises(ValueError):
			self.circular_buffer.remove(100)

		for i in range(self.buffer_size):
			self.circular_buffer.append(i)
		
		self.circular_buffer.remove(5)
		self.assertEqual(self.circular_buffer, [0, 1, 2, 3, 4, 6, 7, 8, 9])
		
		# Test removing element not in buffer
		with self.assertRaises(ValueError):
			self.circular_buffer.remove(20)

	def test_sliced_reads(self):
		for i in range(self.buffer_size):

			self.circular_buffer.append(i)
		
		self.assertEqual(self.circular_buffer[2:5], [2, 3, 4])
		
		# Test sliced read past end
		self.assertEqual(self.circular_buffer[8:12], [8, 9])

	def test_sliced_writes(self):
		for i in range(self.buffer_size):
			self.circular_buffer.append(i)
		
		self.circular_buffer[2:5] = [99, 98, 97]
		self.assertEqual(self.circular_buffer, [0, 1, 99, 98, 97, 5, 6, 7, 8, 9])

		self.circular_buffer[8:12] = [88, 77]
		self.assertEqual(self.circular_buffer, [0, 1, 99, 98, 97, 5, 6, 7, 88, 77])
		
		# Test sliced write past end
		with self.assertRaises(IndexError):
			self.circular_buffer[8:12] = [88, 77, 66, 55]

	def test_move_range(self):
		cases = {
			(0, 0, 0): [1, 2, 3, 4, 5, 6, 7, 8],
			(1, 3, 5): [1, 2, 3, 4, 5, 2, 3, 4],
			(1, 4, 6): [4, 5, 3, 4, 5, 6, 2, 3],
			(6, 4, 2): [1, 2, 7, 8, 1, 2, 7, 8],
			(6, 4, 5): [2, 2, 3, 4, 5, 7, 8, 1],
			(5, 3, 6): [8, 2, 3, 4, 5, 6, 6, 7],
			(0, 8, 0): [1, 2, 3, 4, 5, 6, 7, 8],
			(3, 1, 6): [1, 2, 3, 4, 5, 6, 4, 8],
		}
		for k, v in cases.items():
			self.circular_buffer.fill([1, 2, 3, 4, 5, 6, 7, 8])
			self.circular_buffer._move_range(*k)
			print(self.circular_buffer.buffer)
			self.assertEqual(self.circular_buffer, v)
			print()

	def test_random_inserts_removals(self):
		operations = ['append', 'insert', 'remove', 'set', 'eq', 'ne', 'ge', 'gt', 'le', 'lt', 'rotate', 'add', 'sub', 'mul', 'div', 'pow', 'mod', 'round', 'sort', 'insort', 'clear', 'uniq', 'contains', 'serialise']
		# operations = ["append", "pop", "set", "round"]
		expected = list(self.circular_buffer)
		for _ in range(100000):
			operation = random.choice(operations)
			if operation == 'append':
				x = random.randint(0, 100)
				self.circular_buffer.append(x)
				expected.append(x)
			elif operation == 'pop' and len(self.circular_buffer) > 0:
				pos = random.randint(0, len(self.circular_buffer)-1)
				self.circular_buffer.pop(pos)
				expected.pop(pos)
			elif operation == 'insert':
				pos = random.randint(0, len(self.circular_buffer))
				x = random.randint(0, 100)
				self.circular_buffer.insert(pos, x)
				expected.insert(pos, x)
			elif operation == 'remove' and len(self.circular_buffer) > 0:
				pos = random.randint(0, len(self.circular_buffer)-1)
				x = self.circular_buffer[pos]
				self.circular_buffer.remove(x)
				expected.remove(x)
			elif operation == 'set' and len(self.circular_buffer) > 2:
				pos = random.randint(-len(self.circular_buffer)+1, len(self.circular_buffer)-2)
				if random.randint(0, 1):
					pos += random.random()
				else:
					pos = float(pos)
				x = random.random()
				self.circular_buffer[pos] = x
				if pos == int(pos):
					expected[int(pos)] = x
				else:
					left, right = floor(pos), ceil(pos)
					expected[left] = expected[left] * (pos - left) + x * (right - pos)
					expected[right] = expected[right] * (right - pos) + x * (pos - left)
			elif operation == 'eq':
				assert self.circular_buffer == expected
			elif operation == 'ne':
				assert not self.circular_buffer != expected
			elif operation == 'ge' and not random.randint(0, 10):
				assert all(self.circular_buffer >= expected)
			elif operation == 'gt' and not random.randint(0, 10):
				assert not any(self.circular_buffer > expected)
			elif operation == 'le' and not random.randint(0, 10):
				assert all(self.circular_buffer <= expected)
			elif operation == 'lt' and not random.randint(0, 10):
				assert not any(self.circular_buffer < expected)
			elif operation == 'rotate' and len(self.circular_buffer) > 0:
				pos = random.randint(-len(self.circular_buffer)+1, len(self.circular_buffer)-1)
				pos = 1
				self.circular_buffer.rotate(pos)
				expected = deque(expected)
				expected.rotate(pos)
				expected = list(expected)
			elif operation == 'add':
				self.circular_buffer += 1
				expected = [x + 1 for x in expected]
			elif operation == 'sub':
				self.circular_buffer -= 1
				expected = [x - 1 for x in expected]
			elif operation == 'mul' and not random.randint(0, 10):
				r = random.choice((1 / 1.1, 1.1))
				self.circular_buffer *= r
				expected = [x * r for x in expected]
			elif operation == 'div' and not random.randint(0, 10):
				r = random.choice((1 / 1.1, 1.1))
				self.circular_buffer /= r
				expected = [x / r for x in expected]
			elif operation == 'pow' and not random.randint(0, 10):
				self.circular_buffer **= 1
				expected = [x ** 1 for x in expected]
			elif operation == 'mod':
				self.circular_buffer %= 100
				expected = [x % 100 for x in expected]
			elif operation == 'round':
				n = 0
				self.circular_buffer.round(n)
				expected = [round(x, n) for x in expected]
			elif operation == 'sort' and not random.randint(0, 100):
				self.circular_buffer.sort()
				expected.sort()
			elif operation == 'insort' and not random.randint(0, 100):
				x = random.randint(0, 100)
				self.circular_buffer.insort(x)
				expected.sort()
				bisect.insort_left(expected, x)
			elif operation == 'clear' and not random.randint(0, 1000):
				self.circular_buffer.clear()
				expected.clear()
			elif operation == 'uniq' and not random.randint(0, 100):
				sort = random.choice((True, False, None))
				self.circular_buffer.uniq(sort=sort)
				assert self.circular_buffer == set(expected), self.circular_buffer.symmetric_difference(expected)
				expected = list(self.circular_buffer)
			elif operation == 'contains' and self.circular_buffer:
				pos = random.randint(0, len(self.circular_buffer)-1)
				assert expected[pos] in self.circular_buffer
			elif operation == 'serialise' and not random.randint(0, 100):
				b = pickle.dumps(self.circular_buffer)
				self.circular_buffer = pickle.loads(b)
			assert len(self.circular_buffer) == len(expected), operation
			if not self.circular_buffer:
				continue
			assert self.circular_buffer == expected, operation
			# close = self.circular_buffer.isclose(expected)
			# if np.all(close):
			# 	continue
			# indices = np.nonzero(close == 0)
			# assert False, (operation, self.circular_buffer.shape, indices[0].shape, self.circular_buffer[indices], np.asanyarray(expected, dtype=np.float64)[indices])

	def test_rigorous_inserts_removals(self):
		size = 1000
		for i in range(size):
			self.circular_buffer.append(i)
		self.assertEqual(self.circular_buffer, range(size))
		r = range(-3000, 3000)
		nums = list(r)
		random.shuffle(nums)
		for i in nums:
			self.circular_buffer.insert(random.randint(0, len(self.circular_buffer)), i)
		self.assertEqual(set(self.circular_buffer), set(r))
		random.shuffle(nums)
		for i in nums:
			self.circular_buffer.remove(i)
		self.assertEqual(sorted(self.circular_buffer), list(range(size)))
		
	def test_boundary_conditions(self):
		# Test empty buffer
		self.assertEqual(self.circular_buffer, [])
		
		# Test buffer full then empty
		for i in range(self.buffer_size):
			self.circular_buffer.append(i)
		for i in range(self.buffer_size):
			self.circular_buffer.remove(self.circular_buffer[0])

		self.assertEqual(self.circular_buffer, [])
		
		# Test single element
		self.circular_buffer.append(1)
		self.assertEqual(self.circular_buffer, [1])
		self.circular_buffer.remove(1)
		self.assertEqual(self.circular_buffer, [])


class TestThreadSafeList(unittest.TestCase):

	def setUp(self):
		self.ts_list = ThreadSafeList()

	def test_append_thread_safety(self):
		def append_items():
			for i in range(1000):
				self.ts_list.append(i)

		threads = [threading.Thread(target=append_items) for _ in range(10)]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Verify the list length
		self.assertEqual(len(self.ts_list), 10000)

	def test_read_write_thread_safety(self):
		def append_items():
			for i in range(1000):
				self.ts_list.append(i)

		def read_items():
			for _ in range(1000):
				index = len(self.ts_list) - 1
				if index >= 0:
					_ = self.ts_list[index]

		append_thread = threading.Thread(target=append_items)
		read_thread = threading.Thread(target=read_items)

		append_thread.start()
		read_thread.start()

		append_thread.join()
		read_thread.join()

		# Verify the list length
		self.assertEqual(len(self.ts_list), 1000)

	def test_modify_thread_safety(self):
		for i in range(1000):
			self.ts_list.append(i)

		def modify_items():
			for i in range(1000):
				index = i % len(self.ts_list)
				self.ts_list[index] = i

		threads = [threading.Thread(target=modify_items) for _ in range(10)]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Verify no unexpected exceptions and the list length remains the same
		self.assertEqual(len(self.ts_list), 1000)

	def test_insert_thread_safety(self):
		def insert_items():
			for i in range(500):
				self.ts_list.insert(0, i)

		threads = [threading.Thread(target=insert_items) for _ in range(10)]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Verify the list length
		self.assertEqual(len(self.ts_list), 5000)

	def test_delete_thread_safety(self):
		for i in range(10000):
			self.ts_list.append(i)

		def delete_items():
			for _ in range(700):
				if len(self.ts_list) > 0:
					i = random.randint(0, len(self.ts_list) // 2 - 1) if len(self.ts_list) > 20 else 0
					del self.ts_list[i]

		threads = [threading.Thread(target=delete_items) for _ in range(10)]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Verify the list length does not go below zero
		self.assertGreaterEqual(len(self.ts_list), 0)
		self.assertLessEqual(len(self.ts_list), 10000)


if __name__ == "__main__":
	unittest.main()