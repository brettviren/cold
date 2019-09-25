#!/usr/bin/env python3
'''
Main CLI interface to COLD
'''
import math
import time
import click
import numpy
import matplotlib.pyplot as plt
from collections import namedtuple

from .util import Chirp

@click.group()
@click.pass_context
def cli(ctx):
    pass




@cli.command("check-torch")
def check_torch():
    import torch

    if torch.cuda.is_available():
        ncuda = torch.cuda.device_count()
        click.echo ("have %d CUDA devices:" % ncuda)
        for ind in range(ncuda):
            torch.cuda.device(ind)
            click.echo ('%d: %s' % (ind, torch.cuda.get_device_name()))
    else:
        click.echo ("no CUDA")
    return


@cli.command("check-bee")
@click.option("-d","--device", default="cuda",
              type=click.Choice(['cuda','cpu']),
              help="the 'gpu' device to use")
@click.argument("bee-file")
@click.argument("pdf-file")
def check_bee(device, bee_file, pdf_file):
    '''
    Load a bee file, plot it
    '''
    import torch
    chirp = Chirp("bee: ")
    torch.tensor([0,], device=device)
    chirp('warm up device "%s", reset time' % device)
    chirp.reset()
    from cold import io
    meta, xyzqt = io.load_bee(bee_file,device=device)
    chirp("loaded %s from %s" % (xyzqt[:0].size(), bee_file))

    pts = xyzqt.cpu().numpy()
    chirp("convert to numpy")

    fig, axes = plt.subplots(2,2)
    bins = 100
    axes[0,0].hist2d(pts[:,0], pts[:,1],bins)
    axes[0,1].hist2d(pts[:,0], pts[:,2],bins)
    axes[1,0].hist2d(pts[:,2], pts[:,1],bins)
    axes[1,1].hist(pts[:,3],bins,log=True)
    plt.savefig(pdf_file)
    chirp("done")



@cli.command("test")
@click.option("-d","--device", default="cuda",
              type=click.Choice(['cuda','cpu']),
              help="the 'gpu' device to use")
@click.option("--work-shape", default=(2048, 8192),
              nargs=2,
              help="number of columns/wires and rows/ticks for convolution")
@click.argument("wires-file")
@click.argument("response-file")
@click.argument("bee-file")
def check_test(device, work_shape, wires_file, response_file, bee_file):
    import torch
    chirp = Chirp("stest: ")
    torch.tensor([0,], device=device)
    chirp('warm up device "%s", reset time' % device)
    chirp.reset()

    from cold import io, drift, wires, units, binning, ductor, splat, ifconv, util, bypixel, bydepo

    work_shape = tuple(map(util.fftsize, work_shape))
    print ("work shape: ", work_shape)

    # detector description.  fixme: refactor this to come from a cold.detectors.<name>.Geometry object
    wire_plane = (1,0) # fixme: we just look at one plane now
    chmap = wires.pdsp_channel_map(wires_file, 'cpu')
    all_pimpos = wires.pdsp_pimpos(chmap)
    pimpos = all_pimpos[wire_plane]  
    tbinning = binning.Binning(6000,0,3*units.ms)
    bbs = wires.pdsp_bounds(chmap)
    bb0 = bbs[wire_plane]
    # print ("bb0[m]:",bb0.minp/units.m, bb0.maxp/units.m)
    # print ("pitch[mm]:",
    #         pimpos.region_binning.minedge*units.mm,
    #         pimpos.region_binning.maxedge*units.mm)
    chirp("load 'geometry'")

    # response data
    res = numpy.load(response_file)
    # fixme: we just look at one plane now
    res0 = torch.tensor(res['resp0'], dtype=torch.float, device='cpu') 
    chirp("load response")

    # nodes
    bee = io.BeeFiles(device='cpu')
    drifter = drift.Drifter(respx = pimpos.origin[0] + 10*units.cm, positive_facing=True) # fixme
    pitcher = drift.Pitcher(pimpos)
    tbinner = binning.WidthBinning(tbinning)
    pbinner = binning.WidthBinning(pimpos.region_binning)
    #duct = ductor.Ductor(pimpos, tbinning, res0)
    #splat = splat.Splat(pimpos, tbinning)
    # splat = bypixel.Splat(pimpos, tbinning)
    splat = bydepo.Splat(pimpos, tbinning)
    duct = ifconv.Ductor(pimpos.nimper, res0, work_shape, device=device)
    chirp("make nodes")

    # run graph
    points = bee(bee_file)
    xyzq = torch.stack((points['x'], points['y'], points['z'], points['q'])).T
    #print("all points:",xyzq.shape[0])
    intpc = bb0.inside(xyzq[:,:3])
    xyzq = xyzq[intpc,:]
    #print("points in TPC:",xyzq.shape[0])
    assert xyzq.shape[0] > 0

    x,y,z,q = xyzq.T
    t = torch.zeros_like(x)
    #print ("x[m]:", torch.min(x)/units.m, torch.max(x)/units.m)
    chirp("load points")

    drifted = drifter(x, t, q)
    pitched = pitcher(y,z)

    # print ("Pdrift[mm]:",torch.min(pitched['Pdrift']), torch.max(pitched['Pdrift']))
    # print ("<dP[mm]>:",torch.sum(drifted['dP']*units.mm)/len(drifted['dP']))
    # print ("Tdrift[us]:",torch.min(drifted['Tdrift']), torch.max(drifted['Tdrift']))
    # print ("<dT[us]>:",torch.sum(drifted['dT']*units.us)/len(drifted['dT']))
    chirp("drifted and pitched")

    tbins = tbinner(drifted['Tdrift'], drifted['dT'])
    pbins = pbinner(pitched['Pdrift'], drifted['dP'])
    chirp("binned")


    # splatted = splat(drifted['Qdrift'],
    #                  drifted['Tdrift'], drifted['dT'],
    #                  pitched['Pdrift'], drifted['dP'],
    #                  tbins['bins'], tbins['span'],
    #                  pbins['bins'], pbins['span'])

    splatted = splat(drifted['Qdrift'],
                     drifted['Tdrift'], drifted['dT'],
                     pitched['Pdrift'], drifted['dP'])


    chirp("splatted")

    # ducted = duct(splatted['ion'])

    # chirp("done")







@cli.command("check-drift")
@click.option("-d","--device", default="cuda",
              type=click.Choice(['cuda','cpu']),
              help="the 'gpu' device to use")
@click.argument("bee-file")
@click.argument("pdf-file")
def check_drift(device, bee_file, pdf_file):
    '''
    Load a bee file, plot it
    '''
    chirp = Chirp("drift: ")
    torch.tensor([0,], device=device)
    chirp('warm up device "%s", reset time' % device)
    chirp.reset()
    from cold import io, drift
    meta, xyzqt = io.load_bee(bee_file,device=device)
    chirp("loaded %s from %s" % (xyzqt[:0].size(), bee_file))
    drifted = drift.points(xyzqt)
    chirp("drifted")

    dlong,dtrans,tnew,qnew = drifted.T.cpu().numpy()
    x,y,z,q,t = xyzqt.T.cpu().numpy()
    
    fig, axes = plt.subplots(2,2)
    bins = 100
    axes[0,0].hist(x-tnew, bins)
    axes[1,0].hist2d(tnew, dlong, bins)
    axes[1,1].hist2d(tnew, dtrans, bins)
    axes[0,1].hist2d((q-qnew)/q, x, bins)

    plt.savefig(pdf_file)
    chirp("done")

@cli.command("check-splat")
@click.option("-d","--device", default="cuda",
              type=click.Choice(['cuda','cpu']),
              help="the 'gpu' device to use")
@click.argument("bee-file")
@click.argument("pdf-file")
def check_splat(device, bee_file, pdf_file):
    chirp = Chirp("splat: ")
    torch.tensor([0,], device=device)
    chirp('warm up device "%s", reset time' % device)
    chirp.reset()

    from cold import io, drift, splat, util
    meta, xyzqt = io.load_bee(bee_file,device=device)
    chirp("loaded %s from %s" % (xyzqt[:0].size(), bee_file))

    drifted = drift.points(xyzqt)
    chirp("drifted")

    depos = util.point_drifted_depos(xyzqt, drifted, None)
    chirp("converted")

    # test pitch
    pbounds = torch.tensor((0.0,1000.0))
    pbb = splat.binning_bounds(depos[:,2], 3*depos[:,4], pbounds, 0.5, 10)
    chirp(str(pbb.shape))

    field = torch.zeros((10000,10000), device=device)

    nimperwire = 10

    noob = 0
    # (integral, r_center, c_center, r_sigma, c_sigma)
    for depo in depos:
        orig, patch = splat.full_patch(depo)
        irow,icol = map(int, orig)
        nrow,ncol = map(int, patch.shape)

        if irow < 0 or icol < 0 or irow+nrow >= 10000 or icol+ncol >= 10000:
            noob += 1
            continue

        field[irow:irow+nrow, icol:icol+ncol] += patch

    chirp("out of bounds: %d" % noob)
    plt.imshow(field.cpu())
    plt.colorbar()
    plt.savefig(pdf_file)
    chirp("done")

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
    
