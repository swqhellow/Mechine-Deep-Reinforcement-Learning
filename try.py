import heapq
import numpy as np
np.random.seed(10)
arry = np.random.randint(1, 50, 10)
print(arry)
res = []
heapq.heappush(res, arry)
print(res)
res1 = []
for num in arry:
    heapq.heappush(res1, num)
print(res1)
