#!/usr/bin/env python3
'''
Main CLI interface to COLD
'''
import math
import time
import click
import torch
import numpy
import matplotlib.pyplot as plt
from collections import namedtuple

@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command("check-torch")
def check_torch():
    if torch.cuda.is_available():
        ncuda = torch.cuda.device_count()
        click.echo ("have %d CUDA devices:" % ncuda)
        for ind in range(ncuda):
            torch.cuda.device(ind)
            click.echo ('%d: %s' % (ind, torch.cuda.get_device_name()))
    else:
        click.echo ("no CUDA")
    return

@cli.command("load")
@click.option("-d","--device", default="cuda",
              type=click.Choice(['cuda','cpu']),
              help="the 'gpu' device to use")
@click.argument("npz-file")
def load_test(device, npz_file):
    t0 = time.time()
    def chirp(msg):
        dt = time.time() - t0
        print ("%10.3f: %s" % (dt, msg))

    torch.tensor([0,], device=device)
    chirp ("warm up GPU with single element tensor, reset time")
    t0 = time.time()

    dat = numpy.load(npz_file)
    chirp ("loaded data: %s" % (",".join(dat.keys())), )

    finetick = 0.1              # us
    tick = 0.5                  # us
    nfineticksperadctick = round(tick/finetick)
    nimperwire = 10
    nrespwires = 21
    pitch = 5.0                 # mm
    imp_pitch = pitch/nimperwire
    nsigma = 3                  # bounds on depo
    nticks = 6000 # match 3ms PDSP for now, must be smaller than ncols-resp ticks
    nwires = 1000 # must be smaller than nrows - nrespwires
    max_pitch = nwires*pitch 

    # The convolution space is taken to be large enough to avoid wrap
    # around and rounded up to a higher power of 2 to speed up FFT.
    # The convolution is done on a per impact position basis.
    # wire columns by time ticks.
    nrows = 1024
    ncols = 8196
    work_shape = (nrows, ncols)

    # just do plane 0 in this prototype

    # go to 0.5us ticks
    # 210 total rows, 21 after decimation
    # 200 total columns
    plane_wires = list()
    for iplane in range(3):

        r0fine = dat['resp%d'%iplane]
        chirp ('loaded response')
        r0 = r0fine.reshape((r0fine.shape[0], r0fine.shape[1]//nfineticksperadctick,-1)).sum(axis=1)
        chirp ('rebinned response')
        r0 = torch.tensor(r0, device=device)
        chirp ('loaded to gpu')

        # big array to hold ionization electron raster 
        ion_field = torch.zeros((nwires*nimperwire, nticks), device='cpu')
        chirp('initialized ionization field on cpu')

        # p in mm, t in us
        Depo = namedtuple('Depo','n p t dp dt')
        depos = [
            Depo(5000, 0.0, 0.0, 1.0, 1.0),
            Depo(1000, 1000.0, 1000.0, 3.0, 2.0),
        ]
        for depo in depos:
            p_range = [int(math.floor(max(0, depo.p - nsigma*depo.dp)/imp_pitch)),
                       int(math.ceil(min(max_pitch, depo.p + nsigma*depo.dp)/imp_pitch))]
            t_range = [int(math.floor(max(0, depo.t - nsigma*depo.dt)/tick)),
                       min(nticks, round((depo.t + nsigma*depo.dt)/tick))]
            for imp in range(*p_range): # span transverse/pitch dimension
                p = imp*imp_pitch
                dp = p - depo.p
                n = depo.n * (0.5/depo.dp) * math.exp(-dp*dp/(2*depo.dp*depo.dp))
                for itick in range(*t_range): # span drift/tick dimension
                    t = itick*tick
                    dt = t - depo.t
                    q = n * (0.5/depo.dt) * math.exp(-dt*dt/(2*depo.dt*depo.dt))
                    ion_field[imp, itick] += q
        chirp ("made depos")

        wires = torch.zeros(work_shape, device=device)
        chirp ("made result array on device")

        for imp in range(nimperwire):

            # ionization
            ion_work = torch.zeros(work_shape, device=device)
            ion_byimp = ion_field[imp::nimperwire, :]
            ion_work[:ion_byimp.shape[0], :ion_byimp.shape[1]] = ion_byimp;
            ion_byimp_c = torch.stack((ion_work, torch.zeros_like(ion_work)), 2)
            ion_spec = torch.fft(ion_byimp_c, 2)

            # detector response
            r_byimp = torch.zeros(work_shape, device=device)
            r = r0[imp::10,:]
            r_byimp[:r.shape[0], :r.shape[1]] = r
            r_byimp_c = torch.stack((r_byimp, torch.zeros_like(r_byimp)), 2)
            s_byimp_c = torch.fft(r_byimp_c, 2)

            wires += torch.ifft(s_byimp_c * ion_spec, 2)[:,:,0] # accumulate real part

        plane_wires.append(wires.cpu());

    chirp("done calculation")

    w = plane_wires[0].numpy()
    plt.imshow(w[100:300, 1800:2200])
    plt.colorbar()
    plt.savefig("foo.pdf")
        

def main():
    cli()

if '__main__' == __name__:
    main()
    
