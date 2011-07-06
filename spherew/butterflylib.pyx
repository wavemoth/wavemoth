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

    int bfm_apply_d(char *matrixdata, double *x, double *y,
                    bfm_index_t nrows, bfm_index_t ncols, bfm_index_t nvecs)

    const_char *bfm_scatter_2(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2)
    const_char *bfm_scatter_2(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2)
    const_char *bfm_scatter(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2, int32_t nvecs)
    const_char *bfm_scatternot_2(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2)
    const_char *bfm_scatternot(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2, int32_t nvecs)
    const_char *bfm_scatter_add_2(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2)
    const_char *bfm_scatter_add(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2, int32_t nvecs)
    const_char *bfm_scatternot_add_2(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2)
    const_char *bfm_scatternot_add(
        const_char *mask, 
        double *target1,
        double *target2,
        double *source,
        int len1, int len2, int32_t nvecs)

#
# Wrappers around individual parts for unit testing
#

def scatter(np.ndarray[char, mode='c'] mask,
            np.ndarray[double, ndim=2, mode='c'] target1,
            np.ndarray[double, ndim=2, mode='c'] target2,
            np.ndarray[double, ndim=2, mode='c'] source,
            bint not_mask=False,
            bint add=False,
            int repeat=1):
    cdef int i
    cdef const_char *retval
    cdef char *pm
    cdef double *pt1, *pt2, *ps
    cdef int len1, len2
    
    nvecs = target1.shape[1]
    assert nvecs == target2.shape[1] == source.shape[1]
    if not_mask:
        assert mask.sum() == source.shape[0]
    else:
        assert mask.shape[0] - mask.sum() == source.shape[0]
    assert mask.shape[0] == target1.shape[0] + target2.shape[0]

    pm = <char*>mask.data
    pt1 = <double*>target1.data
    pt2 = <double*>target2.data
    ps = <double*>source.data
    len1 = target1.shape[0]
    len2 = target2.shape[0]

    # Dispatch to all the cases; have the benchmark loop within.
    if nvecs == 2:
        if not_mask:
            if add:
                for i in range(repeat):
                    retval = bfm_scatternot_add_2(pm, pt1, pt2, ps, len1, len2)
            else:
                for i in range(repeat):
                    retval = bfm_scatternot_2(pm, pt1, pt2, ps, len1, len2)
        else:
            if add:
                for i in range(repeat):
                    retval = bfm_scatter_add_2(pm, pt1, pt2, ps, len1, len2)
            else:
                for i in range(repeat):
                    retval = bfm_scatter_2(pm, pt1, pt2, ps, len1, len2)
    else:
        if not_mask:
            if add:
                for i in range(repeat):
                    retval = bfm_scatternot_add(pm, pt1, pt2, ps, len1, len2, nvecs)
            else:
                for i in range(repeat):
                    retval = bfm_scatternot(pm, pt1, pt2, ps, len1, len2, nvecs)
        else:
            if add:
                for i in range(repeat):
                    retval = bfm_scatter_add(pm, pt1, pt2, ps, len1, len2, nvecs)
            else:
                for i in range(repeat):
                    retval = bfm_scatter(pm, pt1, pt2, ps, len1, len2, nvecs)

    assert <char*>mask.data + mask.shape[0] == retval
    return np.vstack([target1, target2])

