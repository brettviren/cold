#!/usr/bin/env python3
'''
Deal with depos.
'''

import os
import time
import numpy
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mydir = os.path.dirname(os.path.realpath(__file__))

class Splat(object):
    '''
    Transform individual 2D Gaussian parameters into a 2D field.
    '''
    nsigma = 3.0

    def __init__(self, pimpos, tbinning, **params):
        self.__dict__.update(**params)

        self.tb = tbinning
        self.pb = pimpos.impact_binning

        self.bindesc = numpy.asarray([self.tb.minedge, self.tb.binsize, 
                                      self.pb.minedge, self.pb.binsize],
                                     dtype=numpy.float32)
        self.shape = (self.pb.nbins, self.tb.nbins)

        # bypixel(float* field, float* bindesc, int* offset, float* depo)
        code = open(os.path.join(mydir,"bydepo.cu")).read()
        self.mod =  SourceModule(code)
        self.meth = self.mod.get_function("bypixel")


    def __call__(self, Qdrift, Tdrift, dT, Pdrift, dP, **kwds):

        depos = numpy.vstack((Qdrift.cpu().numpy(),
                              Tdrift.cpu().numpy(),
                              dT.cpu().numpy(),
                              Pdrift.cpu().numpy(),
                              dP.cpu().numpy()
        )).T.copy(order='C')
        ndepos = depos.shape[0]

        tbeg = self.tb.bin_trunc(Tdrift - self.nsigma*dT).cpu().numpy()
        tend = self.tb.bin_trunc(Tdrift + self.nsigma*dT).cpu().numpy()
        pbeg = self.pb.bin_trunc(Pdrift - self.nsigma*dP).cpu().numpy()
        pend = self.pb.bin_trunc(Pdrift + self.nsigma*dP).cpu().numpy()

        tnum = tend - tbeg + 1
        pnum = pend - pbeg + 1

        Nticks = numpy.zeros(ndepos) + self.shape[1]

        offsets = numpy.vstack((tbeg, pbeg, Nticks))
        offsets = offsets.T.copy(order='C')

        out = numpy.zeros(self.shape, dtype=numpy.float32)

        for idepo in range(ndepos):
            depo = depos[idepo];
            offset = offsets[idepo];
            block=(int(tnum[idepo]), int(pnum[idepo]), 1)
            #print (idepo, block, offset)
            print (idepo, depo)
            self.meth(drv.InOut(out),
                      drv.In(self.bindesc),
                      drv.In(offset),
                      drv.In(depo),
                      block=block)
        t1 = time.time()
        print (t1-t0)
        return out

