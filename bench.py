from spherew.matvec import *

N = 100
A = np.arange(N * N, dtype=np.double).reshape(N, N)
x = np.arange(N, dtype=np.complex)
x = x - 1j * x

y1 = np.dot(A, x)
y2 = dmat_zvec(A, x)

