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
    of shape (4,N) holding N depos as (x,y,z,q).  Note, no time is
    provided by Bee files.
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

