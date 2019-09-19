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

    respx = 0.0*units.mm
    # True if wire plane face points in postive X direction
    positive_facing = True      

    DL=7.2*units.cm**2/units.s
    DT=12.0*units.cm**2/units.s
    lifetime = 8*units.ms
    speed=1.6*units.mm/units.us
    
    # fixme: not yet supported
    # fluctuate = False

    def __init__(self, **params):
        self.__dict__.update(**params)

    def __call__(self, x, t, q, **kwds):

        if self.positive_facing:
            backup = x < self.respx
            driftsign = +1.0
        else:
            backup = x > self.respx
            driftsign = -1.0

        dt = driftsign*(x - self.respx)/self.speed
        tnew = t + dt
        dtabs = torch.abs(dt)

        absorbprob = 1 - torch.exp((-1.0/self.lifetime)*dtabs)
        dQ = q * absorbprob
        # fixme: add fluctuation
        Qf = q-dQ

        dlong = torch.sqrt(2.0*self.DL*dtabs)
        dtran = torch.sqrt(2.0*self.DT*dtabs)

        # except we undo for backups
        Qf[backup] = q[backup]
        dlong[backup] = 0.0
        dtran[backup] = 0.0

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
