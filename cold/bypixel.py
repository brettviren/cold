#!/usr/bin/env python3
'''
A splat method which is per-pixel parallel.
'''

import os
import time
import numpy
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from jinja2 import Template

mydir = os.path.dirname(os.path.realpath(__file__))


class Splat(object):
    nsigma = 3.0

    def __init__(self, pimpos, tbinning, **params):
        self.__dict__.update(**params)

        tb = tbinning
        pb = pimpos.region_binning

        self.shape = (pb.nbins, tb.nbins)
        self.delta = (pb.binsize, tb.binsize)
        self.origin = (pb.minedge, tb.minedge)

        tpl = Template(open(os.path.join(mydir,"bypixel.cu")).read())
        code = tpl.render(
            dPITCH = self.delta[0],
            PITCH0 = self.origin[0],
            NTICKS = self.shape[1],
            dTIME = self.delta[1],
            TIME0 = self.origin[1],
            NSIGMA = self.nsigma);
        self.mod =  SourceModule(code)
        self.meth = self.mod.get_function("bypixel")


    def __call__(self, Qdrift, Tdrift, dT, Pdrift, dP, **kwds):
        ndepo = Qdrift.size()
        ndepo = 100
        ndepo = numpy.asarray(ndepo)
        out = numpy.zeros(self.shape, dtype=numpy.float32)
        t0 = time.time()
        print ("splatting %d depos" % ndepo)

        self.meth(drv.InOut(out),
                  drv.In(Qdrift.cpu().numpy()),
                  drv.In(Pdrift.cpu().numpy()),
                  drv.In(Tdrift.cpu().numpy()),
                  drv.In(dP.cpu().numpy()),
                  drv.In(dT.cpu().numpy()),
                  drv.In(ndepo),
                  grid=(self.shape[0], self.shape[1], 1),
                  block=(1,1,1)
        )
        t1 = time.time()
        print (t1-t0)
        return out

def dotest():

    nticks = 6000
    nwires = 1148


    params = dict(
        NTICKS=nticks,
        dPITCH=0.5,
        dTIME=0.5,
        PITCH0=-2.5,
        TIME0=0.0,
        NSIGMA=3.0)

    tpl = Template(open(os.path.join(mydir,"test.cu")).read())
    code = tpl.render(**params)

    print (code)

    tmod = SourceModule(code)

    # testit(float* out, float* q, float* p, float* t, float* dp, float* dt, int* ndepo)
    testit = tmod.get_function("testit")

    #out = drv.mem_alloc(nwires*nticks*4)
    out = numpy.zeros((nwires,nticks), dtype=numpy.float32)
    q = numpy.asarray([5000], dtype=numpy.float32)
    p = numpy.asarray([-.5], dtype=numpy.float32)
    t = numpy.asarray([2], dtype=numpy.float32)
    dp = numpy.asarray([1], dtype=numpy.float32)
    dt = numpy.asarray([1], dtype=numpy.float32)
    ndepo = numpy.asarray([1], dtype=numpy.int32)

    t0 = time.time()
    testit(drv.InOut(out),
           drv.In(q),
           drv.In(p),
           drv.In(t),
           drv.In(dp),
           drv.In(dt),
           drv.In(ndepo),
           grid=(nwires, nticks, 1),
           block=(1,1,1)
    )

    t1 = time.time()
    print ("%.3fs %s\n%s" % (t1-t0, out.shape, out))




# class Splat(object):
#     '''
#     Raster a set of 2D Gaussian distributions.
#     '''
#     tbinning = None
#     pimpot = None

#     def __init__(self, pimpos, tbinning, **kwds):
#         self.pimpos = pimpos
#         self.tbinning = tbinning
#         self.pls = self.pimpos.impact_binning.linspace()
#         self.tls = self.tbinning.linspace()

#     def raster(self, imp, q, pmean, tmean, psigma, tsigma):
        


#     def __call__(self, q, pmean, tmean, psigma, tsigma, **kwds):

#         for imp in range(self.pimpos.nimper):
#             yield self.raster(imp, q, pmean, tmean, psigma, tsigma)
