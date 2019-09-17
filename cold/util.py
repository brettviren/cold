#!/usr/bin/env python3
'''
Utility functions
'''
import time
import torch

def point_drifted_depos(points, drifted, plane):
    '''
    Given charge point (N,5) (x,y,z,t,q) tensor and drifted (N,4)
    tensor (dL,dT,tnew,qnew) return depo (N,5) tensor (q,p,t,dp,dt).
    '''
    # cheat for now
    
    return torch.cat((
        drifted[:,3].reshape(1,-1), # qnew
        points[:,2].reshape(1,-1),  # p: this part is a cheat for now
        drifted[:,2].reshape(1,-1), # t
        drifted[:,1].reshape(1,-1), # dT: transverse
        drifted[:,0].reshape(1,-1), # dL: longitudinal
    )).T

class Chirp(object):

    def __init__(self, prefix=""):
        self._prefix = prefix
        self.reset()

    def __call__(self, msg):
        now = time.time()
        dt = now - self._t0
        ddt = now - self._tlast
        self._tlast = now
        print ("%s%10.3f (+%.6f): %s" % (self._prefix, dt, ddt, msg))

    def reset(self):
        self._tlast = self._t0 = time.time()
        
