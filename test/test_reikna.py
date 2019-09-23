import numpy as np
from reikna import cluda
from reikna.fft import FFT

def test_fft():
    api = cluda.cuda_api()
    thr = api.Thread.create()

    N = 256
    M = 10000

    #data_in = np.random.rand(N, N) + 1j*np.random.rand(N, N)
    data_in = np.random.rand(N, N).astype('complex')
    cl_data_in = thr.to_device(data_in)
    cl_data_out = thr.empty_like(cl_data_in)
    fft = FFT(thr).prepare_for(cl_data_out, cl_data_in, -1, axes=(0,))

if __name__ == "__main__":
    test_fft()
