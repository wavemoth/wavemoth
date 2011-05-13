cdef extern from "ylmgen_c.h":
    ctypedef struct Ylmgen_C:
        double *ylm
    
    void Ylmgen_init(Ylmgen_C *gen, int l_max, int m_max, int s_max,
                     int spinrec, double epsilon)
    void Ylmgen_set_theta(Ylmgen_C *gen, double *theta, int nth)
    void Ylmgen_destroy (Ylmgen_C *gen)
    void Ylmgen_prepare (Ylmgen_C *gen, int ith, int m)
    void Ylmgen_recalc_Ylm (Ylmgen_C *gen)
    void Ylmgen_recalc_lambda_wx (Ylmgen_C *gen, int spin)
    double *Ylmgen_get_norm (int lmax, int spin, int spinrec)

cimport numpy as np
import numpy as np
from libc.string cimport memcpy

def computed_normalized_associated_legendre(int m, theta,
                                            int lmax, double epsilon=1e-30,
                                            out=None):
    """
    Given a value for m, computes the matrix :math:`\tilde{P}_\ell^m(\theta)`,
    with values for ``theta`` taken along rows and l = 0..l_max along columns.
    
    """
    cdef Ylmgen_C ctx
    cdef Py_ssize_t col, row
    cdef np.ndarray[double, mode='c'] theta_ = np.ascontiguousarray(theta, dtype=np.double)
    cdef np.ndarray[double, ndim=2] out_
    if out is None:
        out = np.empty((theta_.shape[0], lmax + 1), np.double)
    out_ = out
    if out_.shape[0] != theta_.shape[0] or out_.shape[1] != lmax + 1:
        raise ValueError("Invalid shape of out")
    Ylmgen_init(&ctx, lmax, lmax, 0, 0, epsilon)
    try:
        Ylmgen_set_theta(&ctx, <double*>theta_.data, theta_.shape[0])
        for row in range(theta_.shape[0]):
            Ylmgen_prepare(&ctx, row, m)
            Ylmgen_recalc_Ylm(&ctx)
            for col in range(lmax + 1):
                out[row, col] = ctx.ylm[col]
    finally:
        Ylmgen_destroy(&ctx)
    return out
    
    
