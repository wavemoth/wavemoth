from _openmp import *
import contextlib

@contextlib.contextmanager
def use_num_threads(nthreads):
    old_nt = get_max_threads()
    old_dyn = get_dynamic()
    try:
        set_dynamic(False)
        set_num_threads(nthreads)
        yield
    finally:
        set_dynamic(old_dyn)
        set_num_threads(old_nt)
