from queue import Queue
import numpy as np

# array = np.array((4,3))
# print(array)
# print(array.shape)

# expanded_array = np.expand_dims(array, axis=0)

# print(expanded_array)
# print(expanded_array.shape)

q = Queue()

q.put(2)
q.put(0)
q.put(1)
q.put(3)

print(q.qsize())

print(q.get())
print(q.get())
print(q.qsize())
print(q.get())
print(q.get())

if (q.empty()):
    print('e')
