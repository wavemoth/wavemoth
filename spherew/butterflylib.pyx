from __future__ import division
cimport numpy as np
import numpy as np

cdef extern from "malloc.h":
    void *memalign(size_t boundary, size_t size)

cdef extern from "butterfly.h":
    ctypedef int bfm_index_t
    ctypedef int int32_t
    ctypedef int int64_t
    ctypedef char const_char "const char"

    const_char* bfm_scatter(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        size_t len1,
        size_t len2,
        size_t nvecs,
        int group,
        int should_add)

    int bfm_apply_d(char *matrixdata, double *x, double *y,
                    bfm_index_t nrows, bfm_index_t ncols, bfm_index_t nvecs)


#
# Wrappers around individual parts for unit testing
#

def scatter(np.ndarray[char, mode='c'] mask,
            np.ndarray[double, ndim=2, mode='c'] target1,
            np.ndarray[double, ndim=2, mode='c'] target2,
            np.ndarray[double, ndim=2, mode='c'] source,
            int group,
            bint add=False,
            int repeat=1):
    cdef int i
    cdef const_char *retval
    cdef char *pm
    cdef double *pt1, *pt2, *ps
    cdef int len1, len2

    assert group in (0, 1)
    
    nvecs = target1.shape[1]
    assert nvecs == target2.shape[1] == source.shape[1]
    assert (mask == group).sum() == source.shape[0]
    assert mask.shape[0] == target1.shape[0] + target2.shape[0]

    retval = bfm_scatter(<char*>mask.data,
                         <double*>target1.data,
                         <double*>target2.data,
                         <double*>source.data,
                         target1.shape[0],
                         target2.shape[0],
                         nvecs, group, add)
    assert <char*>mask.data + mask.shape[0] == retval
    return np.vstack([target1, target2])

