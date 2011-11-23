from __future__ import division

import numpy as np

def power_of_two(x):
    return 2**int(np.log2(x)) == x

def block_add_reduce(buf, buf_idx):
    buf[buf_idx, :16] = buf[buf_idx, ::2] + buf[buf_idx, 1::2]
    buf[buf_idx, 16:] = buf[buf_idx + 1, ::2] + buf[buf_idx + 1, 1::2]

num_buffers = 0

def warp_tree_reduction(row_start, row_stop, buf, buf_idx, is_top):
    if row_stop - row_start == 1:
        # Leaf -- really: process next row of Lambda and reduce
        # in registers and scratch independent of nvecs to 16
        # in width.
        # here: dummy data
        buf[buf_idx, :] = 1 * row_start
    else:
        nrows = row_stop - row_start
        ncols = 16 // nrows
        warp_tree_reduction(row_start, row_start + nrows // 2, buf, buf_idx, True)
        warp_tree_reduction(row_start + nrows // 2, row_stop, buf, buf_idx + 1, False)
        block_add_reduce(buf, buf_idx)#buf_top, buf_bottom, out)

buf = np.zeros((6, 32))

warp_tree_reduction(0, 16, buf, 0, True)
print buf[0]
print num_buffers
