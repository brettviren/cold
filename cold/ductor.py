#!/usr/bin/env python3
'''
Convolve drifted ionization electron distributions and field response.
'''
import torch
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

    def __call__(self, Qdrift, Tdrift, Pdrift, dT, dP, **kwds):

        rb = self.pimpos.region_binning
        ib = self.pimpos.impact_binning
        tb = self.tbinning

        pmin_bin = ib.bin_trunc(rb.edge(rb.bin(Pdrift - self.nsigma*dP)    ))
        pmax_bin = ib.bin_trunc(rb.edge(rb.bin(Pdrift + self.nsigma*dP) + 1)) + 1
        pnbins = pmax_bin - pmin_bin
        pmin = ib.edge(pmin_bin)
        pmax = ib.edge(pmax_bin)

        tmin_bin = tb.bin_trunc(Tdrift - self.nsigma*dT)
        tmax_bin = tb.bin_trunc(Tdrift + self.nsigma*dT) + 1
        tnbins = tmax_bin - tmin_bin
        tmin = tb.edge(tmin_bin)
        tmax = tb.edge(tmax_bin)

        good = Qdrift > self.minq
        good *= pmin_bin >= 0
        good *= pmax_bin <= rb.nbins
        good *= tmin_bin >= 0
        good *= tmax_bin <= tb.nbins
        good *= pnbins > 1
        good *= tnbins > 1


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
        for imp in range(nimper):

            work_r.zero_()
            work_r[:nreswires, :nresticks] = self.response[imp::nimper]
            work_c.zero_()
            work_c[:,:,0] = work_r
            res_spec = torch.fft(work_c, 2)

            work_r.zero_()
            # Raster each depo impact slice
            for ind,q in enumerate(Qdrift):
                if not good[ind]:
                    continue
                # fixme: split depos into per-apa units
                ip = int(pmin_bin[ind])
                fp = int(pmax_bin[ind])
                it = int(tmin_bin[ind])
                ft = int(tmax_bin[ind])
                if ip < 0 or fp > rb.nbins:
                    print ("pitch out of bounds: %d - %d" % (ip, fp))
                    continue
                if it < 0 or ft > tb.nbins:
                    print ("tick out of bounds: %d - %d" % (it, ft))
                    continue

                pls = torch.linspace(pmin[ind], pmax[ind], pnbins[ind], device=Qdrift.device)[imp::nimper]
                if 0 == len(pls):
                    continue
                tls = torch.linspace(tmin[ind], tmax[ind], tnbins[ind], device=Qdrift.device)
                if 0 == len(tls):
                    continue

                pmg, tmg = torch.meshgrid(pls,tls)
                print ("shapes",imp,ind,pls.shape, tls.shape, pmg.shape,tmg.shape)
                pgauss = gauss(Pdrift[ind], dP[ind], pmg)
                tgauss = gauss(Tdrift[ind], dT[ind], tmg)
                patch = q * pgauss * tgauss
                patch_tot += float(torch.sum(patch))


                np = int(patch.shape[0])
                nt = int(patch.shape[1])
                
                work_r[ip:ip+np, it:it+nt] += patch
                qtot += float(q)

            work_c.zero_()
            work_c[:,:,0] = work_r
            q_spec = torch.fft(work_c, 2) # torch.stack((work_r, torch.zeros_like(work_r)), 2), 2)
            tmp = torch.ifft(res_spec * q_spec, 2)
            volts += tmp[:rb.nbins, :tb.nbins, 0]
            #volts += torch.ifft(res_spec * q_spec, 2)[:rb.nbins,:tb.nbins,0]

            print ("imp:",imp, patch_tot, torch.sum(volts), torch.sum(tmp))



        print ("qtot",qtot,"patch tot",patch_tot)
        print (volts)
        return dict(signals=volts)
