#!/usr/bin/env python3
'''
Convolve drifted ionization electron distributions and field response.
'''
import torch
from time import time
from . import units
from .util import fftsize, gauss

class Ductor(object):

    nimper = 10
    device = 'cuda'
    work_shape = None
    resp_specs = list()
    

    def __init__(self, nimper, resp, work_shape, **params):
        self.nimper = nimper

        # something big enough for (all wires + 21, all ticks + 200)
        # and best to be composed of small prime factors.
        self.work_shape = work_shape
        self.__dict__.update(**params)

        work_r = torch.zeros(work_shape, dtype=torch.float, device=self.device)
        work_c = torch.stack((work_r, torch.zeros_like(work_r)), 2)

        # (210, 200)
        nimps, nticks = resp.shape
        nwires = nimps//nimper   # 21

        #print (nwires, nticks, nimper)

        for imp in range(nimper):
            work_c.zero_()
            work_c[:nwires,:nticks,0] = resp[imp::nimper]
            rr = torch.fft(work_c, 2)
            self.resp_specs.append(rr)



    def __call__(self, ion, **kwds):
        '''
        ion is (nimps, nticks), output is (nwires, nticks)
        '''

        nimps, nticks = ion.shape
        nwires = nimps//self.nimper

        volts = torch.zeros((nwires, nticks), dtype=torch.float, device=self.device)

        #print(self.work_shape)

        work_r = torch.zeros(self.work_shape, dtype=torch.float, device=self.device)
        work_c = torch.stack((work_r, torch.zeros_like(work_r)), 2)

        for imp in range(self.nimper):

            work_c.zero_()
            work_c[:nwires,:nticks,0] = ion[imp::self.nimper,:nticks]
            q_spec = torch.fft(work_c, 2)
            tmp = torch.ifft(self.resp_specs[imp] * q_spec, 2)
            volts += tmp[:nwires, :nticks, 0]

        return dict(signals=volts)
