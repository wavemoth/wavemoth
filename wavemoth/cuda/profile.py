from contextlib import contextmanager
import tempfile
import os
import csv
import numpy as np

from . import flatcuda as cuda

class AttrDict(dict):
    def __getattr__(self, attrname):
        return self[attrname]

class CudaProfile(object):
    def __init__(self):
        self.kernels = {}
        
    def format(self, kernel_name, nflops, nwarps=None):
        stats = self.kernels[kernel_name]
        times = np.asarray(stats['times'])
        dt = np.min(times)

        occupancy_fraction = stats['occupancy'][0]
        occ_warps = int(np.round(occupancy_fraction * 48))

        s = '%.3e +/- %.0e sec = %.2f GFLOP/s, ' % (
            dt, np.std(times), nflops / dt / 1e9)
        if nwarps is not None:
            s += 'occupancy %.2f (%d warps in %d blocks)' % (
                occupancy_fraction, occ_warps, occ_warps // nwarps)
        else:
            s += 'occupancy %.2f (%d warps)' % (
                occupancy_fraction, occ_warps)
        return s

def mktemp():
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    return filename

def parse_profile_csv(filename, result):
    with file(filename) as f:
        lines = list(csv.reader(f))
    for line in lines:
        if line[0].startswith('#') or line[0].startswith('NV_') or line[0] == 'method':
            continue
        if len(line) == 3:
            method, gputime, cputime = line
            occupancy = 0
        elif len(line) == 4:
            method, gputime, cputime, occupancy = line
        if method not in result.kernels:
            stats = result.kernels[method] = {}
        else:
            stats = result.kernels[method]
        for listname, value in [('times', float(gputime) * 1e-6),
                                ('occupancy', float(occupancy))]:
            if listname not in stats:
                stats[listname] = []
            stats[listname].append(value)

@contextmanager
def cuda_profile(*args, **kw):
    options=['method', 'gputime', 'cputime', 'occupancy']
    config_file = mktemp()
    profile_file = mktemp()
    try:
        results = CudaProfile(*args, **kw)
        with file(config_file, 'w') as f:
            f.write('\n'.join(options))
        cuda.initialize_profiler(config_file, profile_file, cuda.CSV)
        cuda.start_profiler()
        try:
            yield results
        finally:
            # The stop_profiler call is important as it waits until
            # asynchronous execution is done
            cuda.stop_profiler()
            parse_profile_csv(profile_file, results)
    finally:
        os.unlink(config_file)
        os.unlink(profile_file)
