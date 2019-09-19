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

        qtot = 0.0
        patch_tot = 0.0
        nqless = 0

        # 10
        nimper = self.pimpos.nimper 
        # 21
        nreswires = round(self.response.shape[0] / nimper)
        # 200
        nresticks = int(self.response.shape[1])

        t_raster = 0
        t_conv1 = 0
        t_conv2 = 0

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

                pls = torch.linspace(pmin[ind], pmax[ind], nimper*pnbins[ind], device=Qdrift.device)[imp::nimper]
                if 0 == len(pls):
                    continue
                tls = torch.linspace(tmin[ind], tmax[ind], tnbins[ind], device=Qdrift.device)
                if 0 == len(tls):
                    continue

                pmg, tmg = torch.meshgrid(pls,tls)
                pgauss = gauss(Pdrift[ind], dP[ind], pmg)
                tgauss = gauss(Tdrift[ind], dT[ind], tmg)
                patch = q * pgauss * tgauss
                patch_tot += float(torch.sum(patch))

                ip = int(pbin0[ind])
                it = int(tbin0[ind])
                np = int(patch.shape[0])
                nt = int(patch.shape[1])
                
                work_r[ip:ip+np, it:it+nt] += patch
                qtot += float(q)

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

            print ("imp:",imp, patch_tot, torch.sum(volts), torch.sum(tmp))



        print ("qtot",qtot,"patch tot",patch_tot)
        print ("raster:",t_raster,"t_conv1",t_conv1,"t_conv2",t_conv2)
        return dict(signals=volts)
