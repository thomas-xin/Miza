import time, collections
from smath import *

        
def compare(size):
    print("Creating...")
    
    _list = list(range(size))
    _deque = collections.deque(range(size))
    _hlist = hlist(range(size))


    t = time.time()
    _list.append(0)
    a1 = time.time() - t

    t = time.time()
    _deque.append(0)
    a2 = time.time() - t

    t = time.time()
    _hlist.append(0)
    a3 = time.time() - t

    print("Append at end: " + str(a1) + ", " + str(a2) + ", " + str(a3))


    t = time.time()
    _list.insert(0, 0)
    a1 = time.time() - t

    t = time.time()
    _deque.appendleft(0)
    a2 = time.time() - t

    t = time.time()
    _hlist.appendleft(0)
    a3 = time.time() - t

    print("Append at start: " + str(a1) + ", " + str(a2) + ", " + str(a3))


    t = time.time()
    _list.insert(size >> 2, 0)
    a1 = time.time() - t

    t = time.time()
    _deque.insert(size >> 2, 0)
    a2 = time.time() - t

    t = time.time()
    _hlist.insert(size >> 2, 0)
    a3 = time.time() - t

    print("Append at 1/4: " + str(a1) + ", " + str(a2) + ", " + str(a3))


    t = time.time()
    _list.insert(size >> 1, 0)
    a1 = time.time() - t

    t = time.time()
    _deque.insert(size >> 1, 0)
    a2 = time.time() - t

    t = time.time()
    _hlist.insert(size >> 1, 0)
    a3 = time.time() - t

    print("Append at 1/2: " + str(a1) + ", " + str(a2) + ", " + str(a3))


    t = time.time()
    _list.insert(size * 3 >> 2, 0)
    a1 = time.time() - t

    t = time.time()
    _deque.insert(size * 3 >> 2, 0)
    a2 = time.time() - t

    t = time.time()
    _hlist.insert(size * 3 >> 2, 0)
    a3 = time.time() - t

    print("Append at 3/4: " + str(a1) + ", " + str(a2) + ", " + str(a3))


    t = time.time()
    len(_list)
    a1 = time.time() - t

    t = time.time()
    len(_deque)
    a2 = time.time() - t

    t = time.time()
    len(_hlist)
    a3 = time.time() - t

    print("Count elements: " + str(a1) + ", " + str(a2) + ", " + str(a3))


    t = time.time()
    _list[size >> 1]
    a1 = time.time() - t

    t = time.time()
    _deque[size >> 1]
    a2 = time.time() - t

    t = time.time()
    _hlist[size >> 1]
    a3 = time.time() - t

    print("Get middle element: " + str(a1) + ", " + str(a2) + ", " + str(a3))


    t = time.time()
    del _list
    a1 = time.time() - t

    t = time.time()
    del _deque
    a2 = time.time() - t

    t = time.time()
    del _hlist
    a3 = time.time() - t

    print("Delete: " + str(a1) + ", " + str(a2) + ", " + str(a3))
    

if __name__ == "__main__":
    while True:
        compare(int(input("Enter iterable size: ")))
