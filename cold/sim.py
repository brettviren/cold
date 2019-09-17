#!/usr/bin/env python3
'''
LArTPC simulation
'''

import cold.drift
import cold.splat

def points_to_waves(xyzqt):
    '''
    Take in (N,5) tensor of point charges, return waveform
    '''

    # (dlong,dtrans,tnew,qnew).
    drifted = cold.drift.points(xyzqt)

    
