#!/usr/bin/env python3
'''
Raster 2D Gaussian distributions.
'''

import math
from time import time
from collections import defaultdict
from .util import gauss
import torch



class Timer(object):
    def __init__(self):
        self.times = defaultdict(float)
        self.start()
    def start(self):
        self.t0 = time()
    def __call__(self, key):
        now = time()
        self.times[key] += now-self.t0
        self.t0 = now
    def __str__(self):
        l = list()
        for k,t in sorted(self.times.items()):
            l.append("%20s: %0.4f" % (k,t))
        return '\n'.join(l)


class Splat(object):

    tbinning = None
    pbinning = None

    def __init__(self, pimpos, tbinning, **params):
        self.tbinning = tbinning
        self.pimpos = pimpos
        self.__dict__.update(**params)

    def __call__(self, Qdrift,
                 Tdrift, dT,
                 Pdrift, dP, 
                 tbins, tspan,
                 pbins, pspan, **kwds):

        nimper = self.pimpos.nimper 

        tmin,tmax = tspan.T
        it, nt = tbins.T

        pmin,pmax = pspan.T
        pbin0,pnbins = pbins.T

        ip = pbin0 * nimper
        np = pnbins * nimper

        t_ls = 0.0
        t_gaus = 0.0
        t_bc = 0.0
        t_sum = 0.0
        
        timer = Timer()

        pls = self.pimpos.impact_binning.linspace(device=Qdrift.device)
        tls = self.tbinning.linspace(device=Qdrift.device)
        
        ion = torch.zeros((self.pimpos.impact_binning.nbins, self.tbinning.nbins),
                          dtype=torch.float, device=Qdrift.device)
        #print ("ion array shape:", ion.shape)
        timer("init")

        for ind, q in enumerate(Qdrift):


            this_ip = ip[ind]
            this_it = it[ind]

            this_np = np[ind]
            this_nt = nt[ind]

            pgauss = gauss(Pdrift[ind], dP[ind], pls[this_ip:this_ip+this_np])
            tgauss = gauss(Tdrift[ind], dT[ind], tls[this_it:this_it+this_nt]) * q
            timer("gauss")

            patch = torch.broadcast_tensors(pgauss.view(this_np, 1), tgauss.view(1, this_nt))
            timer("patch")

            patch = patch[0] * patch[1]
            timer("mult")
            # pgauss = gauss(Pdrift[ind], dP[ind], pmg[this_ip:this_ip+this_np,this_it:this_it+this_nt])
            # tgauss = gauss(Tdrift[ind], dT[ind], tmg[this_ip:this_ip+this_np,this_it:this_it+this_nt])
            # patch = q * pgauss * tgauss

            ion[this_ip:this_ip+this_np,this_it:this_it+this_nt] += patch

            timer("sum")
            

        print (timer)
        return dict(ion=ion)






def binning_bounds(center, halfwidth, bounds, binsize, mod=1):
    '''
    Given 1D tensors "center" and "halfwidth" and scalar "binsize",
    return a tensor (N,2) holding bin bounds at least big enough and
    enlarged to the given modulo boundary but no bigger than bounds a
    2-element tensor.
    '''
    # edges relative to bounds
    m = center-halfwidth-bounds[0]
    p = center+halfwidth-bounds[0]
    # discretize as per modulo
    delta = binsize*mod
    m = ((m / delta).type(torch.int) * delta).type(torch.float)
    p = ((p / delta).type(torch.int) * delta).type(torch.float)
    p += delta
    m = torch.max(m, bounds[0].expand_as(m))
    p = torch.min(p, bounds[1].expand_as(p))
    
    return torch.cat((m.reshape(1,-1),
                      p.reshape(1,-1)))
    

def linspace(center, halfwidth, binsize, mod=1, device='cuda'):
    '''
    Return a "linspace" which covers at least center +/- halfwidth and
    possibly expanded to reach bin%mod == 0.
    '''
    bin0 = round((center-halfwidth)/binsize)
    binf = round((center+halfwidth)/binsize)
    bin0 -= bin0%mod
    binf += mod - binf%mod - 1
    return torch.linspace(bin0*binsize, binf*binsize, binf-bin0+1, device=device)


def gauss(mean, sigma, mg):
    '''
    Calculate an exponential across the meshgrid
    '''
    rel = (mg-mean)/sigma
    return 0.5*sigma * torch.exp(-0.5*rel*rel);



def full_patch(blip, r_bin=0.5, c_bin=500, nsigma=3, r_mod=10, c_mod=1):
    '''
    Raster a 2D gaussian bound by a number of sigma.

    The "blip" object should be a 5-element 1D tensor:

    (integral, r_center, c_center, r_sigma, c_sigma)

    Where "r" and "c" stand for the row and column dimension,
    respectively.  Eg, for a "g" representing an energy deposition:

    (nelectrons, pitch, time, pitch_sigma, time_sigma).

    The r_bin and c_bin gives the 2D bin sizes.

    The nsigma limits the size of the output tensor relative to the
    Gaussian extent.

    The r_mod and c_mod sets how much beyond nsigma to fill so that
    the resulting patch terminates at r_bin%r_mod == 0 row boundaries,
    etc for columns.

    Return value is a pair.

    (offset, patch)

    The offset is a tuple giving the number of bins to the start of
    the patch and the patchis a 2D tensor with the rastered blip.
    '''
    mag, r_mean, c_mean, r_sigma, c_sigma = map(float, blip)
    r_ls = linspace(r_mean, nsigma*r_sigma, r_bin, r_mod, blip.device)
    c_ls = linspace(c_mean, nsigma*c_sigma, c_bin, c_mod, blip.device)


    r_mg, c_mg = torch.meshgrid(r_ls, c_ls)
    r_gauss = gauss(r_mean, r_sigma, r_mg)
    c_gauss = gauss(c_mean, c_sigma, c_mg)
    the_patch = mag * r_gauss * c_gauss

    return ((int(r_mg[0][0]/r_bin),int(c_mg[0][0]/c_bin)), the_patch)

    # plt.clf(); plt.pcolor(r_mg.cpu(), r_mg.cpu(), patch.cpu()); plt.colorbar()

def fluctuate(field):
    '''
    Return an array of same shape as input which has been
    Poison-randomly fluctuated preserving integral.
    '''
    # fixme: implement!
    return field

