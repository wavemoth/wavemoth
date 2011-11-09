from contextlib import contextmanager
import tempfile
import os
import csv

from . import flatcuda as cuda

class AttrDict(dict):
    def __getattr__(self, attrname):
        return self[attrname]

def mktemp():
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    return filename

def parse_profile_csv(filename, results):
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
        if method not in results:
            results[method] = stats = AttrDict()
        else:
            stats = results[method]
        for listname, value in [('times', float(gputime)),
                                ('occupancy', float(occupancy))]:
            if listname not in stats:
                stats[listname] = []
            stats[listname].append(value)

@contextmanager
def cuda_profile():
    options=['method', 'gputime', 'cputime', 'occupancy']
    config_file = mktemp()
    profile_file = mktemp()
    try:
        results = AttrDict()
        with file(config_file, 'w') as f:
            f.write('\n'.join(options))
        cuda.initialize_profiler(config_file, profile_file, cuda.CSV)
        cuda.start_profiler()
        try:
            yield results
        finally:
            cuda.stop_profiler()
            parse_profile_csv(profile_file, results)
    finally:
        os.unlink(config_file)
        os.unlink(profile_file)
