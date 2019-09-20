#!/usr/bin/env python3
'''
Things related to binnings.
'''
import numpy
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

    def linspace(self, device=None):
        if device is None:
            return numpy.linspace(self.minedge, self.maxedge, self.nbins+1)
        return torch.linspace(self.minedge, self.maxedge, self.nbins+1, device=device)

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
        bounds = torch.tensor([0,self.nbins-1], device=pos.device).type(torch.int)
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


        
class WidthBinning(object):
    '''
    A callable to compute the bin and span of something with a width
    and a mean.  Both ends of the the span is enlarge to the nearest
    bin edges.  Result is constrained to be inside the binnning.
    '''

    binning = None
    nsigma = 3.0

    def __init__(self, binning, **kwds):
        self.binning = binning
        self.__dict__.update(**kwds)

    def __call__(self, means, halfwidths, **kwds):
        "Return dictionary of 'span' and 'bins' tensors of shape (N,2)"
        min_bin = self.binning.bin_trunc(means - self.nsigma*halfwidths)
        max_bin = self.binning.bin_trunc(means + self.nsigma*halfwidths)
        return dict(span = torch.stack((self.binning.edge(min_bin), self.binning.edge(max_bin+1))).T,
                    bins = torch.stack((min_bin, max_bin - min_bin + 1)).T)
        
# class DualWidthBinning(object):
#     '''
#     A callable to compute the bin and span of something with a width
#     and a mean.  The span is enlarge to the edge of the nearest coarse
#     bin edges and the reult is returned in terms of the fine edges.
#     Result is constrained to be inside the fine binnning.
#     '''

#     fb = None
#     cb = None
#     nsigma = 3.0

#     def __init__(self, fine_binning, coarse_binning, **kwds):
#         self.fb = fine_binning
#         self.cb = coarse_binning
#         self.__dict__.update(**kwds)

#     def __call__(self, means, halfwidths, **kwds):
#         "Return dictionary of 'span' and 'bins' tensors of shape (N,2)"
#         min_bin = self.fb.bin_trunc(self.cb.edge(self.cb.bin(means - self.nsigma*halfwidths)))
#         max_bin = self.fb.bin_trunc(self.cb.edge(self.cb.bin(means + self.nsigma*halfwidths) + 1)) + 1
#         return dict(span = torch.stack((self.fb.edge(min_bin),self.fb.edge(max_bin+1))).T,
#                     bins = torch.stack((min_bin, max_bin - min_bin + 1)).T)
        
