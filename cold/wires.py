#!/usr/bin/env python3
'''
Operations with wires.
'''
import torch
def norm(t):
    m2 = torch.sum(t*t)
    if m2 <= 0.0:
        return t
    return t / torch.sqrt(m2)

def make_wires(pitch, angle, origin, device='cuda'):
    '''
    Return a wires tensor of shape (2,2,2)
    '''
    return None

def pitch_direction(wires):
    centers = 0.5(wires[:,0,:] + wires[:,1,:])
    wdir = norm(wires[0][1] - wires[0][0])
    wdir3 = torch.cat(wdir, torch.tensor([0.0]))
    zee = torch.tensor([0.0, 0.0, 1.0])
    pitch = torch.cross(wdir3,zee)
    pdir = norm(pitch[:2])
    return pdir


def pitch_position(xy, pdir, origin=None):
    '''
    Take a tensor of shape (N,2) with columns (x,y) and a tensor of
    shape (2,2,2) with dimensions corresponding to (wire#, tail/head,
    x/y).  That is, wires[0][1] is the head (x,y) point of wire0.

    All x,y points are in a common global coordinate system.  For use
    with the usual 3D Cartesian convention of LArTPC, "x" is
    associated with "detector Z axis" and "y" is "detector Y axis".

    Return a tensor of shape (N) with the pitch position relative to wire[0].
    '''
    if origin is None:
        origin = torch.zeros(2, device=xy.device)

    rel = xy-origin
    return torch.mul(rel, pdir).sum(1)
    
