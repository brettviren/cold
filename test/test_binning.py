#!/usr/bin/env python3

import torch

from cold.wires import Binning

nbins=10
minedge=-50.0
maxedge=50.0
b = Binning(10,-50,50) 

some_indices = [-1,  0,  1,  9, 10, 11]
some_indices_inside = [False, True, True, True, False, False]
some_positions = [-51.0, -50.0, -49.9,  49.9,  50.0,  51.0]

def test_construction():
    assert(b.nbins == nbins)
    assert(b.minedge == minedge)
    assert(type(b.minedge) == float)
    assert(b.maxedge == maxedge)
    assert(type(b.maxedge) == float)
    assert(b.span == maxedge-minedge)

def test_scalar():

    for ind,want in zip(some_indices, some_indices_inside):
        assert(want == b.inbounds(ind))
    try:
        b.inbounds(6.9)
    except TypeError:
        pass
    else:
        raise
        
    try:
        b.inside(0)
    except TypeError:
        pass
    else:
        raise

def test_tensor():
    indices = torch.tensor(some_indices)
    assert(not indices.is_floating_point())
    want = torch.tensor(some_indices_inside)
    got = b.inbounds(indices)
    assert (torch.all(torch.eq(got,want)))

    try:
        b.inside(indices)
    except TypeError:
        pass
    else:
        raise



