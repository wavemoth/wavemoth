import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from scikits.cuda.fft import *
import numpy as np

from wavemoth.cuda.profile import cuda_profile

N = 4092

x = np.asarray(np.random.rand(N), np.float32)
xf = np.fft.fft(x)
x_gpu = gpuarray.to_gpu(x)
xf_gpu = gpuarray.empty(N/2+1, np.complex64)


from wavemoth.cuda import cufft

if 0:
    plan = Plan(x.shape, np.float32, np.complex64)

    from time import time

    t0 = time()
    with cuda_profile() as prof:
        for i in range(1):
            fft(x_gpu, xf_gpu, plan)
    dt = time() - t0
    print 'Python time', dt

    dt = 0
    for kernel, stats in prof.kernels.iteritems():
        dt += sum(stats['times'])
    print dt

    np.allclose(xf[0:N/2+1], xf_gpu.get(), atol=1e-6)



    
