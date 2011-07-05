from time import clock
from contextlib import contextmanager

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

@contextmanager
def benchmark(name, divisor, profile=False):
    if profile:
        import yep
        yep.start("profiles/%s.prof" % name)
    t0 = clock()
    yield
    t1 = clock()
    print '%s, %d iterations: %s per iteration' % (name, divisor, ftime((t1 - t0) / divisor))
    if profile:
        yep.stop()
