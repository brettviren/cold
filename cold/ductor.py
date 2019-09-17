#!/usr/bin/env python3
'''
Convolve drifted ionization electron distributions and field response.
'''
import torch
from . import units

class Ductor(object):

    pimpos = None
    tbinning = None
    nsigma = 3

    def __init__(self, pimpos, tbinning, **params):
        self.pimpos = pimpos
        self.tbinning = tbinning
        self.__dict__.update(**params)

    def __call__(self, Qdrift, Tdrift, Pdrift, dT, dP, **kwds):

        rb = self.pimpos.region_binning
        ib = self.pimpos.impact_binning
        tb = self.tbinning

        pmin_bin = ib.bin_trunc(rb.edge(rb.bin(Pdrift - self.nsigma*dP)    ))
        pmax_bin = ib.bin_trunc(rb.edge(rb.bin(Pdrift + self.nsigma*dP) + 1)) + 1
        pnbins = pmax_bin - pmin_bin
        pmin = ib.edge(pmin_bin)
        pmax = ib.edge(pmax_bin)
        del(pmin_bin)
        del(pmax_bin)

        tmin_bin = tb.bin_trunc(Tdrift - self.nsigma*dT)
        tmax_bin = tb.bin_trunc(Tdrift + self.nsigma*dT) + 1
        tnbins = tmax_bin - tmin_bin
        tmin = tb.edge(tmin_bin)
        tmax = tb.edge(tmax_bin)
        del(tmin_bin)
        del(tmax_bin)

        #bydepo = torch.stack((Qdrift, Tdrift, Pdrift, dT, dP, tnbins, tmin, tmax, pnbins, pmin, pmax)).T

        qtot = 0.0
        # now the painful part.....
        for imp in range(self.pimpos.nimper):
            for ind,q in enumerate(Qdrift):
                qtot += q
        print (qtot)
