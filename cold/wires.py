#!/usr/bin/env python3
'''
Operations with wires.
'''
import math
from collections import defaultdict
import torch
from .binning import Binning
from . import units

def norm(t):
    m2 = t.dot(t)
    if m2 <= 0.0:
        return t
    return t / torch.sqrt(m2)


class Pimpos(object):
    '''
    Plane impact position.

    This class provides an indexed geometric context for a wire plane.

    The indexing is provided through two linear grids which partition
    the pitch direction of the wire plane into fine ("impacts") and
    coarse ("regions" aka "wires") bins.  There are an integral number
    of fine bins spanning a coarse bin which is set to 10 to match the
    11 impact positions per wire region traditionally used for
    Garfield field response simulations.

    Bins of both granularity are numbered by integers starting from
    zero.  The pitch locaion of bin edges are determined relative to
    an origin point expressed in some global coordinate system.  The
    two grids are specified by total number of wires (bin centers of
    "regions") and the boundary wires expressed as a pair of 3D
    endpoints.

    Pictorially, a wire region bin spans between the bin edges marked
    by a "," and centered on a "|" and each is speated into 10 impact
    bins.

    , . . . . | . . . . , . . . . | . . . . , ~~~ , . . . . | . . . . , 
    '''
    def __init__(self, nwires, endwires, origin=None, nimper=10): 
        '''
        nwires is number of wires in this plane

        endwires is tensor (2 wires, 2 endpoints, 3 Cartesian
        coordinates) with endwires[0] giving endpoints of wire 0 and
        endwires[1] giving wire endwires-1

        origin is a 3D point from which all vectors in the Pimpos
        coordinate system are relative.
        '''
        w0, wL = endwires
        self.axis = torch.zeros(9, dtype=endwires.dtype, device=endwires.device).reshape((3,3))
        self.axis[0][0] = 1.0;  # X-axis
        self.axis[1] = norm(w0[1]-w0[0]) # Y-axis is wire axis
        self.axis[2] = torch.cross(self.axis[0], self.axis[1])

        w0c = 0.5*(w0[0] + w0[1])
        wLc = 0.5*(wL[0] + wL[1])

        if origin is None:
            origin = w0c
        self.origin = origin

        pmin = self.pitch_position(w0c)
        pmax = self.pitch_position(wLc)
        hp = 0.5*abs(pmax-pmin)/(nwires-1)

        # use these to do bin queries
        self.region_binning = Binning(nwires,        pmin-hp, pmax+hp)
        self.impact_binning = Binning(nwires*nimper, pmin-hp, pmax+hp)
        self.nimper = nimper

    def transform(self, points):
        '''
        Transform an (N,3) tensor expressed in global into (N,3) in
        pimpos coordinates.  A single point tensor of shape (3) works
        too.
        '''
        v = points - self.origin
        w = torch.sum(v * self.axis[1], points.dim()-1)
        p = torch.sum(v * self.axis[2], points.dim()-1)
        v[:,1] = w
        v[:,2] = p
        return v


    def pitch_position(self, points):
        v = points - self.origin
        return torch.sum(v * self.axis[2], points.dim()-1)
        


# detector description.  fixme: refactor this to come from a cold.detectors.<name>.Geometry object


def pdsp_channel_map(filename, device):
    '''
    Read in a PDSP wire dump file with columns:

    # chan tpc plane wire sx sy sz ex ey ez

    Return dictionary maping

    (tpc,plane,wire)->(channel, wire_ray)
    '''
    channel_map = dict()
    for line in open(filename).readlines():
        line=line.strip()
        if line.startswith("#"):
            continue
        parts = line.split()
        channel = int(parts[0])
        key = tuple(map(int, parts[1:4]))
        ray = torch.tensor(list(map(float, parts[4:])), device=device, dtype=torch.float).reshape((2,3))
        channel_map[key] = (channel,ray*units.cm)
    return channel_map

def pdsp_pimpos(chmap):
    '''
    Return dict mapping:

    (tpc,plane) -> pimpos
    '''
    ret = dict()
    for tpc in range(12):
        for plane,mm in enumerate([(0,1147), (0,1147), (0,479)]):
            w0 = chmap[(tpc,plane,mm[0])][1]
            wL = chmap[(tpc,plane,mm[1])][1]
            ret[(tpc,plane)] = Pimpos(mm[1]+1, torch.stack((w0,wL)))
    return ret

class BoundingBox(object):
    def __init__(self):
        self.minp = None
        self.maxp = None

    def add_ray(self, ray):
        'Add ray tensor of shape (2,3) to bounding box'
        self.add_point(ray[0])
        self.add_point(ray[1])
    def add_point(self, pt):
        'Add pt tensor of shape (3) to bounding box'
        if self.minp is None:
            self.minp = pt
            self.maxp = pt
            return
        self.minp = torch.min(self.minp, pt)
        self.maxp = torch.max(self.maxp, pt)

    def center(self):
        return 0.5*(self.minp+self.maxp)
        
    def inside(self, xyz):
        '''
        Return Boolean tensor of shape (N) for xyz tensor of shape (N,3)
        '''
        return ((self.minp <= xyz) * (xyz <= self.maxp)).all(1)

def pdsp_bounds(chmap):
    '''
    Return dict mapping:

    (tpc,plane) -> boundingbox
    '''
    ret = defaultdict(BoundingBox)

    # (tpc,plane,wire)->(channel, wire_ray)
    for (t,p,w), (ch,ray) in chmap.items():
        ret[(t,p)].add_ray(ray)
    # map TPC number to cathode x position.  This is taken the
    # "volumes" attribute from WCT params.jsonnet for PDSP.
    cathodes = {
        1: -25.4*units.mm,
        2: +25.4*units.mm,
        5: -25.4*units.mm,
        6: +25.4*units.mm,
        9: -25.4*units.mm,
        10: +25.4*units.mm,
    }
    for tpc,cat in cathodes.items():
        for p in range(3):
            c = ret[(tpc,p)].center()
            c[0] = cat
            ret[(tpc,p)].add_point(c)
        
        
    return ret
    
