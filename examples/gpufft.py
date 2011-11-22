print 'initing gpu card'
import pycuda.autoinit
print 'done'
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from scikits.cuda.fft import *
import numpy as np
from time import time

from wavemoth.cuda.profile import cuda_profile

N = 4092

nside = 2048
npix = 12*2048**2
nrings = 4 * nside - 1
lmax = 2 * nside

#x = np.asarray(np.random.rand(N), np.float32)
#xf = np.fft.fft(x)
#x_gpu = gpuarray.to_gpu(x)
#xf_gpu = gpuarray.empty(N/2+1, np.complex64)

map = drv.pagelocked_zeros(npix, np.float64)
buf = drv.pagelocked_zeros((nrings, (lmax + 1) // 2 + 1), np.complex128)

map_gpu = drv.mem_alloc(npix * 8)
buf_gpu = drv.mem_alloc(nrings * ((lmax + 1) // 2 + 1) * 16)

drv.memcpy_htod(map_gpu, map)

from wavemoth.cuda import cufft

print 'ctoring plan'
plan = cufft.HealpixCuFFTPlan(2048, 1)

repeats = 1
print 'plan ctored'
with cuda_profile() as prof:
    t0 = time()
    for i in range(repeats):
        plan.execute(map_gpu, buf_gpu)
    dt = time() - t0
    print dt / repeats
print 'benchmark done'    
#print prof.kernels
dt = 0
for kernel, stats in prof.kernels.iteritems():
    dt += sum(stats['times'])
print dt

drv.memcpy_dtoh(buf, buf_gpu)

print buf[0, :8]
del plan

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



    
