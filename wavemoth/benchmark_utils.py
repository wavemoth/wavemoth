from ._openmp import get_wtime
from contextlib import contextmanager
import os

__all__ = ['ftime', 'benchmark']

def ftime(t):
    if t < 1e-06:
        units = "ns"
        t *= 1e9
    elif t < 1e-03:
        units = "us"
        t *= 1e6
    elif t < 1:
        units = "ms"
        t *= 1e3
    else:
        units = "s"
    return "%.1f %s" % (t, units)

def benchmark(func, repeat, name=None, profile=False, duration=1.0,
              burnin=True):
    if name is None:
        name = func.__name__
    if profile:
        import yep
        yep.start("profiles/%s.prof" % name)
    times = []
    elapsed = 0
    n = 0
    if burnin:
        func(1)
    while elapsed < duration:
        t0 = get_wtime()
        func(repeat)
        t1 = get_wtime()
        elapsed += t1 - t0
        times.append(t1 - t0)
        n += 1
    print '%s, %d x %d: %s' % (name, n, repeat, ftime(min(times) / repeat))
    if profile:
        yep.stop()
    return min(times) / repeat
