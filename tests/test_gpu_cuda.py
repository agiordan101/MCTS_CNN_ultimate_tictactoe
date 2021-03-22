from numba import jit, cuda, njit
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer    
import math

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(10000 / threadsperblock[0])
blockspergrid_y = math.ceil(10000 / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

# normal function to run on cpu 
def func(a):                                 
    for i in range(100000000): 
        a[i]+= 1      
  
# function optimized to run on gpu  
@cuda.jit('void(float64[:])')                    
def func2(a): 
    for i in range(100000000): 
        a[i]+= 1

@njit()                    
def func3(a): 
    for i in range(100000000): 
        a[i]+= 1


if __name__=="__main__": 
    n = 100000000                            
    a = np.ones(n, dtype = np.float64) 
    b = np.ones(n, dtype = np.float32) 
      
    start = timer() 
    func3(a)
    print("with njit:", timer()-start)     
      
    start = timer() 
    func2[blockspergrid, threadsperblock](a) 
    print("with GPU:", timer()-start)
