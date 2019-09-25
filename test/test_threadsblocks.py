import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>

    __global__ void say_hi(float* val)
    {
      int thid = (threadIdx.x+threadIdx.y*blockDim.x+(blockIdx.x*blockDim.x*blockDim.y)+(blockIdx.y*blockDim.x*blockDim.y));
      printf("I have %f and am %dth thread in threadIdx.x:%d.threadIdx.y:%d  blockIdx.:%d blockIdx.y:%d blockDim.x:%d blockDim.y:%d\\n",val[thid], thid,threadIdx.x, threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
    }
    """)

def test_threadsblocks():
    arr = numpy.arange(4*4*2*2).astype(numpy.float32)

    func = mod.get_function("say_hi")
    func(cuda.In(arr), block=(4,4,1),grid=(2,2,1))

    
