#!/usr/bin/env python3
'''
Raster 2D Gaussian distributions.
'''

import math
import torch

def linspace(center, halfwidth, binsize, mod=1, device='cuda'):
    '''
    Return a "linspace" which covers at least center +/- halfwidth and
    possibly expanded to reach bin%mod == 0.
    '''
    bin0 = round((center-halfwidth)/binsize)
    binf = round((center+halfwidth)/binsize)
    bin0 -= bin0%mod
    binf += mod - binf%mod
    return torch.linspace(bin0*binsize, binf*binsize, 1+(binf-bin0), device=device)


def gauss(mean, sigma, mg):
    '''
    Calculate an exponential across the meshgrid
    '''
    rel = (mg-mean)/sigma
    return 0.5*sigma * torch.exp(-0.5*rel*rel);


def patch(blip, r_bin=0.5, c_bin=500, nsigma=3, r_mod=10, c_mod=10, device='cuda'):
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

    The device should be 'cpu' or 'cuda'.

    Return value is a triple of 2D arrays all the same shape:

    (r_mg, c_mg, patch)

    The r_mg and c_mg are "meshgrids" for rows and columns,
    respectively and "patch" is the rastered gaussian.
    '''
    mag, r_mean, c_mean, r_sigma, c_sigma = blip
    r_ls = linspace(r_mean, nsigma*r_sigma, r_bin, r_mod, device)
    c_ls = linspace(c_mean, nsigma*c_sigma, c_bin, c_mod, device)
    r_mg, c_mg = torch.meshgrid(r_ls, c_ls)
    r_gauss = gauss(r_mean, r_sigma, r_mg)
    c_gauss = gauss(c_mean, c_sigma, c_mg)
    the_patch = mag * r_gauss * c_gauss
    return r_mg, c_mg, the_patch

    # plt.clf(); plt.pcolor(r_mg.cpu(), r_mg.cpu(), patch.cpu()); plt.colorbar()

def fluctuate(field):
    '''
    Return an array of same shape as input which has been
    Poison-randomly fluctuated preserving integral.
    '''
    # fixme: implement!
    return field
