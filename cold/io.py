#!/usr/bin/env python3
'''
Routines for dealing with I/O
'''

import json
import torch
from . import units

def load_bee(filename, charge_key='q', device="cuda"):
    '''
    Load a bee JSON file.

    Return tuple (dict, tensor)

    The dict holds scalar attribtes from the file and the tensor is 2D
    of shape (5,N) holding N depos as (x,y,z,q,t).  Note, no time is
    provided by Bee files but the "t" column of zeros is added.
    '''
    dat = json.load(open(filename))
    ret = list()
    for key in ['x','y','z',charge_key]:
        ret.append(dat[key])
        dat.pop(key)

    for dnw in ['q', 'nq']:
        try:
            dat.pop(dnw)
        except KeyError:
            pass


    t = torch.tensor(ret, device=device)*units.cm

    # add an zero array for time:
    z = torch.zeros(t.shape[1], device=device)
    z = z.reshape((1, z.shape[0]))
    t = torch.cat((t, z))

    return (dat, t.T)

class BeeFiles(object):
    charge_key = 'q'
    device = 'cuda'
    dtype = torch.float

    def __init__(self, **params):
        self.__dict__.update(**params)

    def __call__(self, filename):
        dat = json.load(open(filename))
        ret = dict()
        for key in 'xyz':
            ret[key] = units.cm*torch.tensor(dat[key],
                                             device=self.device,
                                             dtype=self.dtype)
        ret['q'] = torch.tensor(dat[self.charge_key],
                                device=self.device,
                                dtype=self.dtype)
        return ret
