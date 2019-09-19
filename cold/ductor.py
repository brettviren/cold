#!/usr/bin/env python3
'''
Convolve drifted ionization electron distributions and field response.
'''
import torch
from time import time
from . import units
from .util import fftsize, gauss

class Ductor(object):

    pimpos = None
    tbinning = None
    response = None
    nsigma = 3
    minq = 0.0                  # protect against zero charge depos

    def __init__(self, pimpos, tbinning, response, **params):
        self.pimpos = pimpos
        self.tbinning = tbinning
        self.response = response
        self.__dict__.update(**params)

    def __call__(self, Qdrift,
                 Tdrift, dT,
                 Pdrift, dP, 
                 tbins, tspan,
                 pbins, pspan, **kwds):

        rb = self.pimpos.region_binning
        ib = self.pimpos.impact_binning
        tb = self.tbinning

        tbin0,tnbins = tbins.T
        tmin,tmax = tspan.T

        pbin0,pnbins = pbins.T
        pmin,pmax = pspan.T

        volts = torch.zeros((rb.nbins,tb.nbins), dtype=torch.float, device=Qdrift.device)

        work_shape = (fftsize(self.response.shape[0] + rb.nbins),
                      fftsize(self.response.shape[1] + tb.nbins))
        print(work_shape)
        work_r = torch.zeros(work_shape, dtype=torch.float, device=Qdrift.device)
        work_c = torch.stack((work_r, torch.zeros_like(work_r)), 2)

        # 10
        nimper = self.pimpos.nimper 
        # 21
        nreswires = round(self.response.shape[0] / nimper)
        # 200
        nresticks = int(self.response.shape[1])

        t_raster = 0
        t_conv1 = 0
        t_conv2 = 0
        t_ls = 0
        t_sum = 0
        t_mg = 0
        t_gauss = 0
        t_patch = 0

        for imp in range(nimper):

            t0 = time()

            work_r.zero_()
            work_r[:nreswires, :nresticks] = self.response[imp::nimper]
            work_c.zero_()
            work_c[:,:,0] = work_r
            res_spec = torch.fft(work_c, 2)

            t1 = time()
            t_conv1 += t1-t0
            t0 = t1

            work_r.zero_()
            # Raster each depo impact slice
            for ind,q in enumerate(Qdrift):

                t0 = time()

                pls = torch.linspace(pmin[ind], pmax[ind], nimper*pnbins[ind], device=Qdrift.device)[imp::nimper]
                if 0 == len(pls):
                    continue
                tls = torch.linspace(tmin[ind], tmax[ind], tnbins[ind], device=Qdrift.device)
                if 0 == len(tls):
                    continue

                t1 = time()
                t_ls += t1-t0
                t0 = t1

                pmg, tmg = torch.meshgrid(pls,tls)

                t1 = time()
                t_mg += t1-t0
                t0 = t1

                pgauss = gauss(Pdrift[ind], dP[ind], pmg)
                tgauss = gauss(Tdrift[ind], dT[ind], tmg)

                t1 = time()
                t_gauss += t1-t0
                t0 = t1

                patch = q * pgauss * tgauss

                t1 = time()
                t_patch += t1-t0
                t0 = t1
                
                work_r[pbin0[ind]:pbin0[ind]+patch.shape[0],
                       tbin0[ind]:tbin0[ind]+patch.shape[1]] += patch

                t1 = time()
                t_sum += t1-t0
                t0 = t1

            t1 = time()
            t_raster += t1-t0
            t0 = t1

            work_c.zero_()
            work_c[:,:,0] = work_r
            q_spec = torch.fft(work_c, 2) # torch.stack((work_r, torch.zeros_like(work_r)), 2), 2)
            tmp = torch.ifft(res_spec * q_spec, 2)
            volts += tmp[:rb.nbins, :tb.nbins, 0]

            t1 = time()
            t_conv2 += t1-t0

            print ("imp:",imp, torch.sum(volts), torch.sum(tmp))


        print ("times: conv1:%.6f conv2:%.6f ls:%.6f mg:%.6f gauss:%.6f patch:%.6f sum:%.6f" %
               (t_conv1, t_conv2, t_ls, t_mg, t_gauss, t_patch, t_sum))

        return dict(signals=volts)
