import numpy as np
cimport numpy as np

cdef extern:
    void iddp_id "iddp_id_"(double *eps, int *m, int *n,
                            double *a, int *krank, int *ilist,
                            double *rnorms)
    void idd_reconint "idd_reconint_"(int *n,
                                      int *ilist,
                                      int *krank,
                                      double *proj,
                                      double *p)

def inverse_permutation(perm):
    invp = np.empty_like(perm)
    invp[perm] = np.arange(len(perm))
    return invp

def sparse_interpolative_decomposition(np.ndarray[double, ndim=2] A,
                                       double eps=1e-10):
    """ Compute the interpolative decomposition of a matrix.

    This version of the routine returns a sparser matrix of the
    form::

        A = A_k [ I_k A_ip ] P

    Returns
    -------

    (iden_list, ipol_list, A_ip)

    idlist: 1D integer array
    iplist: 1D integer array
    A_intp: 2D double array
    """

    cdef int m, n, krank
    cdef np.ndarray[double, ndim=2, mode='fortran'] buf
    cdef np.ndarray[int, mode='fortran'] ilist
    cdef np.ndarray[double, mode='fortran'] rnorms
    buf = A.copy('F')
        
    m = buf.shape[0]
    n = buf.shape[1]
    ilist = np.zeros(n, dtype=np.intc)
    rnorms = np.zeros(n, dtype=np.double)
    iddp_id(&eps, &m, &n, <double*>buf.data,
            &krank, <int*>ilist.data, <double*>rnorms.data)
    out = buf.reshape(n * m, order='F')
    out = out[:krank * (n - krank)]
    out = out.reshape(krank, n - krank, order='F')
    ilist -= 1

    iden_list = ilist[:krank]
    ipol_list = ilist[krank:]

    A_ip = np.empty((krank, n - krank), np.double)
    if n - krank > 0:
        # Permute the output interpolative array to get a more contiguous
        # ipol_list (= P in the following). We have our data as
        # ``A_ip P x``, i.e., ipol_list specifies P as a reordering of
        # rows of ``x``. To permute columns of ``A_ip`` instead, we need
        # to transpose/invert the permutation.
        perm = np.argsort(ipol_list)
        A_ip[:, inverse_permutation(perm)] = out
        ipol_list.sort()
    # Produce the columns in A_k by embedding the permutation
    # as well.
    A_k = np.empty((m, krank), np.double)
    if krank > 0:
        perm = np.argsort(iden_list)
        iperm = inverse_permutation(perm)
        A_k[:, iperm] = A[:, iden_list]
        iden_list.sort()
    return iden_list, ipol_list, A_k, A_ip

def interpolative_decomposition(A, eps=1e-10):
    """ Compute the interpolative decomposition of a matrix.

    Returns
    -------

    ilist, 
    """
    iden_list, ipol_list, A_k, A_ip = sparse_interpolative_decomposition(A, eps)
    k = iden_list.shape[0]
    n = A.shape[1]
    A_ip_full = np.empty((k, n), np.double)
    A_ip_full[:, iden_list] = np.eye(k)
    A_ip_full[:, ipol_list] = A_ip
    return A_k, A_ip_full
