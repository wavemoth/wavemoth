import numpy as np
cimport numpy as np

cdef extern:
    void iddp_id "iddp_id_"(double *eps, int *m, int *n,
                            double *a, int *krank, int *list,
                            double *rnorms)
    
def interpolative_decomposition(double eps, np.ndarray[double, ndim=2,
                                                       mode='fortran'] A):
    """
c       eps -- relative precision of the resulting ID
c       m -- first dimension of a
c       n -- second dimension of a, as well as the dimension required
c            of list
c       a -- matrix to be ID'd
c
c       output:
c       a -- the first krank*(n-krank) elements of a constitute
c            the krank x (n-krank) interpolation matrix proj
c       krank -- numerical rank
c       list -- list of the indices of the krank columns of a
c               through which the other columns of a are expressed;
c               also, list describes the permutation of proj
c               required to reconstruct a as indicated in (*) above
c       rnorms -- absolute values of the entries on the diagonal
c                 of the triangular matrix used to compute the ID
c                 (these may be used to check the stability of the ID)
    """
    
    cdef int m, n, krank
    cdef np.ndarray[int, mode='fortran'] ilist
    cdef np.ndarray[double, mode='fortran'] rnorms
    m = A.shape[0]
    n = A.shape[1]
    ilist = np.empty(n, dtype=np.intc)
    rnorms = np.empty(n, dtype=np.double)
    
    iddp_id(&eps, &m, &n, <double*>A.data,
            &krank, <int*>ilist.data, <double*>rnorms.data)


    return A[:krank, :n - krank], krank, ilist, rnorms
#   c       a -- the first krank*(n-krank) elements of a constitute
#c            the krank x (n-krank) interpolation matrix proj
