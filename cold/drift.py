#!/usr/bin/env python
'''
Implement Wire-Cell drifting algorithm
'''
from . import units
import torch
defaults = dict(
    DL=7.2*units.cm**2/units.s,
    DT=12.0*units.cm**2/units.s,
    lifetime = 8*units.ms,
    speed=1.6*units.mm/units.us,
)

def points(xyzqt, respx=0*units.mm, fluctuate=True, **params):
    '''
    Drift point charges in tensor x along anti-X direction to produce
    2D Gaussian distributions at given location "x".

    Point tensor is assumed to be shaped (N,5) with columns holding
    (x,y,z,q,t) in system of units.

    Return a new tensor shaped (N,4) with the columns holding
    (dlong,dtrans,tnew,qnew).

    Note, the input tensor is not modified but the caller should
    understand that the x coordinates for any future use of the
    drifted points should now be "respx".
    '''
    params = dict(defaults, **params)

    x = xyzqt[:,0]
    Qi = xyzqt[:,3]
    t = xyzqt[:,4]

    # dt can be negative if the point must "back up" to the respx.
    dt = (respx - x)/params['speed']; 
    tnew = t + dt
    dtabs = torch.abs(dt)

    absorbprob = 1 - torch.exp((-1.0/params['lifetime'])*dtabs)
    dQ = Qi * absorbprob
    # fixme: add fluctuation
    Qf = Qi-dQ

    dL = torch.sqrt(2.0*params['DL']*dtabs)
    dT = torch.sqrt(2.0*params['DT']*dtabs)

    n = dL.shape[0]
    return torch.cat((dL.reshape(1,-1),
                      dT.reshape(1,-1),
                      tnew.reshape(1,-1),
                      Qf.reshape(1,-1))).T
    

def plot(depos):
    plt.plot(depos[:,2].cpu(), depos[:,1].cpu(), '.')
    plt.plot(depos[:,2].cpu(), depos[:,0].cpu(), '.')
    
