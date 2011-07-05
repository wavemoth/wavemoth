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

    const_char *post_interpolation_ "post_interpolation"(char *mask, 
                          double *target1,
                          double *target2,
                          double *a,
                          double *b,
                          int len1, int len2, int nvecs)
    const_char *post_interpolation_2(char *mask, 
                                  double *target1,
                                  double *target2,
                                  double *a,
                                  double *b,
                                  int len1, int len2)

#
# Wrappers around individual parts for unit testing
#
def do_post_interpolation(np.ndarray[char, mode='c'] mask,
                          np.ndarray[double, ndim=2, mode='c'] target1,
                          np.ndarray[double, ndim=2, mode='c'] target2,
                          np.ndarray[double, ndim=2, mode='c'] a,
                          np.ndarray[double, ndim=2, mode='c'] b,
                          int repeat=1):
    cdef int i
    cdef const_char *retval
    nvecs = target1.shape[1]
    assert nvecs == target2.shape[1] == a.shape[1] == b.shape[1]
    if not (a.shape[0] + b.shape[0] - 2 == target1.shape[0] + target2.shape[0]):
        raise ValueError("Need scratch space at end of a/b buffers")
    assert mask.sum() == a.shape[0] - 1
    assert mask.shape[0] - mask.sum() == a.shape[0] - 1
    assert mask.shape[0] == target1.shape[0] + target2.shape[0]

    if nvecs == 2:
        for i in range(repeat):
            retval = post_interpolation_2(<char*>mask.data,
                                          <double*>target1.data, <double*>target2.data,
                                          <double*>a.data, <double*>b.data,
                                          target1.shape[0], target2.shape[0])
    else:
        for i in range(repeat):
            retval = post_interpolation_(<char*>mask.data,
                                         <double*>target1.data, <double*>target2.data,
                                         <double*>a.data, <double*>b.data,
                                         target1.shape[0], target2.shape[0],
                                         nvecs)
    return <char*>mask.data + mask.shape[0] - retval

