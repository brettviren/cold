#!/usr/bin/env python3
'''
Do some PyCUDA testing
'''
from __future__ import absolute_import
import time
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

from numba import jit
import numpy


import os
import psutil
process = psutil.Process(os.getpid())

from pycuda.elementwise import ElementwiseKernel
gpu_func = ElementwiseKernel(
        "float *m, float *s, float *g",
        "g[i] = gausser(m[i], s[i])",
        "gaussian_raster",
        preamble="""
        __device__ float gausser(float m, float s)
        { 
          if (s == 0.0) { return 0.0; }
          return s * exp(-0.5*m/s);
        }
        """)
def cpu_func(m, s):
    r = m/s
    c = 1.0/s
    return c * numpy.exp(-0.5*r*r)

@jit(nopython=True)
def numba_func(m, s):
    r = m/s
    c = 1.0/s
    return c * numpy.exp(-0.5*r*r)

def test_hello_gpu():
    print('rss[M]:',process.memory_info().rss/1000000)

    nx = 5000
    ny = 5000
    shape=(nx,ny)
    npoints = nx*ny

    t0 = time.time()
    a = numpy.random.randn(npoints).astype(numpy.float32).reshape(shape)
    b = numpy.random.randn(npoints).astype(numpy.float32).reshape(shape)
    t1 = time.time()
    print ("allocate[kHz]: %.3f, time[us]: %.3f" % (npoints/(t1-t0)/1000, (t1-t0)*1000))

    print('rss[M]:',process.memory_info().rss/1000000)

    t0 = time.time()
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.empty_like(a_gpu)
    gpu_func(a_gpu,b_gpu,c_gpu)
    t1 = time.time()
    print ("gpu speed[kHz]: %.3f, time[us]: %.3f" % (npoints/(t1-t0)/1000, (t1-t0)*1000))

    print('rss[M]:',process.memory_info().rss/1000000)

    t0 = time.time()
    c= cpu_func(a,b)
    t1 = time.time()
    print ("cpu speed[kHz]: %.3f, time[us]: %.3f" % (npoints/(t1-t0)/1000, (t1-t0)*1000))

    t0 = time.time()
    c= numba_func(a,b)
    t1 = time.time()
    print ("numba speed[kHz]: %.3f, time[us]: %.3f" % (npoints/(t1-t0)/1000, (t1-t0)*1000))

    t0 = time.time()
    c= numba_func(a,b)
    t1 = time.time()
    print ("numba2 speed[kHz]: %.3f, time[us]: %.3f" % (npoints/(t1-t0)/1000, (t1-t0)*1000))

    #print (dest-a*b)
    #absum = numpy.sum(numpy.abs(dest-a*b))
    #print (c_gpu)
    #assert (absum == 0.0)
    print('rss[M]:',process.memory_info().rss/1000000)

if '__main__' == __name__:
    test_hello_gpu()
    
