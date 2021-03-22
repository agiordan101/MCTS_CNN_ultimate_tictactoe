import numpy as np
import time
from numba import njit, prange,cuda
import timeit
import numba.cuda


x_cpu = np.ones((10000, 10000), dtype='float32')
out_cpu = np.zeros((10000, 10000), dtype='float32')


@njit(parallel=True, fastmath=True)
def smooth_cpu(x, out_cpu):

    for i in prange(1,x.shape[0]-1):
        for j in range(1,x.shape[1]-1):
            out_cpu[i, j] =  (x[i - 1, j - 1] + x[i - 1, j] + x[i - 1, j + 1] + x[i    , j - 1] + x[i    , j] + x[i    , j + 1] +x[i + 1, j - 1] + x[i + 1, j] + x[i + 1, j + 1]) / 9

# run on cpu
start_time = time.time()
smooth_cpu(x_cpu, out_cpu) # compile before measuring time
print("CPU Time: {0:1.6f}s ".format(time.time() - start_time))
for i in range(10):
    start_time = time.time()
    smooth_cpu(x_cpu, out_cpu)
    print("CPU Time: {0:1.6f}s ".format(time.time() - start_time))
