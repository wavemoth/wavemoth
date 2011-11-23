from __future__ import division

import numpy as np

def power_of_two(x):
    return 2**int(np.log2(x)) == x

def block_add_reduce(top, bottom, output):
    top, top_name = top
    bottom, bottom_name = bottom
    output, output_name = output
    print 'Reducing %s and %s to %s' % (top_name, bottom_name, output_name)
    
    assert np.prod(top.shape) == 32
    assert np.prod(bottom.shape) == 32
    assert np.prod(output.shape) == 32
    h = top.shape[1]
    tmp = top[:, :, ::2] + top[:, :, 1::2] # do not overwrite yet, store in "registers"
    output[:, h:, :] = bottom[:, :, ::2] + bottom[:, :, 1::2]
    output[:, :h, :] = tmp

num_buffers = 0

def allocate(nrows):
    global num_buffers
    num_buffers += 1
    #print '+num_buffers=', num_buffers
    return (np.zeros((2, nrows, 16 // nrows)), 'buf%d' % num_buffers)

def deallocate(arr):
    global num_buffers
    arr[0][...] = np.nan
    num_buffers -= 1
    #print '-num_buffers=', num_buffers

def warp_tree_reduction(row_start, row_stop, out, is_top):
    if row_stop - row_start == 1:
        # Leaf -- really: process next row of Lambda and reduce
        # in registers and scratch independent of nvecs to 16
        # in width.
        # here: dummy data
        out[0][...] = 1 * row_start
    else:
        nrows = row_stop - row_start
        ncols = 16 // nrows

        buf_top = (out[0].reshape((2, nrows // 2, 16 // (nrows // 2))), out[1])
        warp_tree_reduction(row_start, row_start + nrows // 2, buf_top, True)

        buf_bottom = allocate(nrows // 2)
        warp_tree_reduction(row_start + nrows // 2, row_stop, buf_bottom, False)
        block_add_reduce(buf_top, buf_bottom, out)
        
        deallocate(buf_bottom)
        return out
        

r = allocate(16)
warp_tree_reduction(0, 16, r, True)
print r[0][:, :, :].T
print num_buffers

#def append_line(block):
#    assert block.shape == (2, 1, 16)

#for iline in range(32):
#    block = np.ones((2, 1, 16) * iline)
#    append_line(block)
