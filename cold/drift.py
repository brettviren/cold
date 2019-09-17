#!/usr/bin/env python
'''
Implement Wire-Cell drifting algorithm
'''
from . import units
import torch

class Drifter(object):
    '''
    A callable that drifts point charges
    '''

    respx = 0*units.mm
    DL=7.2*units.cm**2/units.s
    DT=12.0*units.cm**2/units.s
    lifetime = 8*units.ms
    speed=1.6*units.mm/units.us
    
    # fixme: not yet supported
    # fluctuate = False

    def __init__(self, **params):
        self.__dict__.update(**params)

    def __call__(self, x, t, q, **kwds):

        dt = (self.respx - x)/self.speed
        tnew = t + dt
        dtabs = torch.abs(dt)

        absorbprob = 1 - torch.exp((-1.0/self.lifetime)*dtabs)
        dQ = q * absorbprob
        # fixme: add fluctuation
        Qf = q-dQ

        dlong = torch.sqrt(2.0*self.DL*dtabs)
        dtran = torch.sqrt(2.0*self.DT*dtabs)

        return dict(dT=dlong, dP=dtran, Qdrift=Qf, Tdrift=tnew)
    
class Pitcher(object):
    '''
    Transform transverse coordinates into a pitch 
    '''

    pimpos = None

    def __init__(self, pimpos, **params):
        self.pimpos = pimpos
        self.__dict__.update(**params)

    def __call__(self, y, z, **kwds):
        x = torch.zeros_like(y)
        r = torch.stack((x,y,z))
        return dict(Pdrift=self.pimpos.pitch_position(r.T))
