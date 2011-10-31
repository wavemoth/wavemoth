from __future__ import division

# Investigate the behaviour of stopping the compression at various
# points -- how much does the residual matrix compress in each step?

from wavemoth import *
from wavemoth.healpix import *
from cmb import as_matrix
from wavemoth.butterfly import butterfly_compress, format_numbytes, InnerNode, IdentityNode
from wavemoth.butterfly import permutations_to_filter
from wavemoth.interpolative_decomposition import sparse_interpolative_decomposition, lssolve

from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt

Nside = 256
lmax = 2 * Nside
epsilon = 1e-30
m = 0

nodes = get_ring_thetas(Nside, positive_only=True)

P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon).T

B = P[:100, :120]
B_even = B[::2, :]
B_odd = B[1::2, :]

#as_matrix(P[::2, :]).plot()
#as_matrix(P[1::2, :]).plot()
 
iden_list1, ipol_list1, A_ip1 = sparse_interpolative_decomposition(B_even, 1e-13)
#iden_list2, ipol_list2, A_ip2 = sparse_interpolative_decomposition(B[1::2, :], 1e-13)

filter1 = permutations_to_filter(iden_list1, ipol_list1).astype(np.bool)
print filter1

#filter1[...] = False

S = lssolve(B_even, iden_list1)

as_matrix(A_ip1).plot()
as_matrix(S).plot()

B_k = B_even[:, ~filter1]
B_recon = np.dot(B_k, A_ip1)
B_other = B[:, filter1]

as_matrix(B_other).plot()

print np.linalg.norm(B_even[:, filter1] - B_recon)

#as_matrix(np.log(np.abs(A_ip1))).plot()
#as_matrix(np.log(np.abs(C))).plot()

#as_matrix(np.dot(B_odd[:, filter1], S) - B_odd[:, ~filter1]).plot()

#as_matrix(np.log(np.abs(A_ip2))).plot()
#as_matrix(A_ip1[:-1, :] - A_ip2[:,:-1]).plot()

#plt.plot(iden_list1, 'o')
#plt.plot(iden_list2, 'o')
#plt.plot(ipol_list1, 'o')
#plt.plot(ipol_list2, 'o')
