from __future__ import division

import numpy as np

def power_of_two(x):
    return 2**int(np.log2(x)) == x

def block_add_reduce(top, bottom, output):
    assert np.prod(top.shape) == 32
    assert np.prod(bottom.shape) == 32
    assert np.prod(output.shape) == 32
    h = top.shape[1]
    output[:, :h, :] = top[:, :, ::2] + top[:, :, 1::2]
    output[:, h:, :] = bottom[:, :, ::2] + bottom[:, :, 1::2]

num_buffers = 0

def allocate(nrows):
    global num_buffers
    num_buffers += 1
    print 'num_buffers=', num_buffers
    return np.zeros((2, 2, nrows, 16 // nrows))

def deallocate(arr):
    global num_buffers
    arr[...] = np.nan
    num_buffers -= 1

def warp_tree_reduction(row_start, row_stop, out):
    if row_stop - row_start == 1:
        # Leaf -- really: process next row of Lambda and reduce
        # in registers and scratch independent of nvecs to 16
        # in width.
        # here: dummy data
        out[...] = 1 * row_start
    else:
        nrows = row_stop - row_start
        ncols = 16 // nrows
        buf = allocate(nrows // 2)
        top_block, bottom_block = buf
        warp_tree_reduction(row_start, row_start + nrows // 2, top_block)
        warp_tree_reduction(row_start + nrows // 2, row_stop, bottom_block)
        block_add_reduce(top_block, bottom_block, out)
        deallocate(buf)
        return out
        

r = np.zeros((2, 16, 1))
warp_tree_reduction(0, 16, r)
print r.T
print num_buffers

#def append_line(block):
#    assert block.shape == (2, 1, 16)

#for iline in range(32):
#    block = np.ones((2, 1, 16) * iline)
#    append_line(block)
