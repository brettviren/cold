#!/usr/bin/env python3
'''
Things related to binnings.
'''

import torch
class Binning(object):
    '''
    Provide histogram-like binning functionality (not actual histogram storage).
    '''
    def __init__(self, nbins, minedge, maxedge):
        self.nbins = nbins
        self.minedge = float(minedge)
        self.maxedge = float(maxedge)

    @property
    def span(self):
        return self.maxedge - self.minedge

    @property
    def binsize(self):
        return self.span / self.nbins

    def inside(self, pos):
        if type(pos) == float:
            return self.minedge <= pos and pos < self.maxedge
        if type(pos) == int or not pos.is_floating_point():
            raise TypeError("Binning.inside() requires float, not int")
        return torch.ge(pos, self.minedge) * torch.lt(pos, self.maxedge)

    def inbounds(self, ind):
        if type(ind) == int:
            return 0 <= ind and ind < self.nbins
        if type(ind) == float or ind.is_floating_point():
            raise TypeError("Binning.inbounds() requires int, not float")
        return torch.ge(ind, 0) * torch.lt(ind,self.nbins)

    def bin(self, pos):
        'Return the bin containing the position(s)'
        if type(pos) == float:
            return int(math.floor((pos - self.minedge) / self.binsize))
        if type(pos) == int:
            raise TypeError("Binning.bin() requires float, not int")
        ndims = pos.dim()-1
        return ((pos - self.minedge) / self.binsize).type(torch.int)

    def bin_trunc(self, pos):
        'Return the bin containing the position(s) truncating to be in the binning'
        bounds = torch.tensor([0,self.nbins-1]).type(torch.int)
        full = self.bin(pos)
        full = torch.max(full, bounds[0].expand_as(full))
        full = torch.min(full, bounds[1].expand_as(full))
        return full

    def center(self, ind):
        'Return the center position of bin index "ind"'
        if type(ind) == int:
            return self.minedge + (ind+0.5) * self.binsize
        if type(ind) == float or ind.is_floating_point():
            raise TypeError("Binning.center() requires int, not float")
        find = ind.type(torch.float)
        return self.minedge + (find+0.5) * self.binsize

    def edge(self, ind):
        'Return the (lower) edge position of bin index "ind"'
        if type(ind) == int:
            return self.minedge + ind * self.binsize
        if type(ind) == float or ind.is_floating_point():
            raise TypeError("Binning.center() requires int, not float")
        find = ind.type(torch.float)
        return self.minedge + find * self.binsize


        
