#
# Imports from pyopencl that we want -- "flat is better than nested"
#

from pyopencl import *
from pyopencl.array import *

READ_WRITE = mem_flags.READ_WRITE
READ_ONLY = mem_flags.READ_ONLY
COPY_HOST_PTR = mem_flags.COPY_HOST_PTR
WRITE_ONLY = mem_flags.WRITE_ONLY

