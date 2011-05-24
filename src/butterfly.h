/*!
Code for applying a butteryfly-compressed matrix to a vector.

TODO: Also contains hard-coded support for Associated Legendre
recursion.


\section Binary format

Consult the paper for the following discussion to make sense. Very
briefly, a butterfly matrix consists of "D S_maxlevel S_{maxlevel-1}
... S_2 S_1", where "D" is a block-diagonal matrix, and "S_l" are
interpolation matrices.

For best temporal and spatial locality, these are best stored in
depth-first manner, in the order the algorithm will traverse the
data. Blocks within each matrix are considered nodes in a tree, with,
e.g., the children of a block of S_3 being in S_2. There are three
different node types; the root node (containing both D and the
top-level S), inner nodes, and leaf nodes.

Each node (=matrix block) consists of three parts: a) The heights of
the sub-blocks it consists of ("row interface"), b) the child nodes
(matrices to the right of itself whose result we depend on), c)
payload used for processing in this matrix block. The sub-block widths
("col interface") is read from the row interface of the children.

TOP-LEVEL COMPRESSED MATRIX FORMAT (HEADER):

 - bfm_index_t: order == 2**level. This gives number of blocks in D.
 - The root node

INNER INTERPOLATION NODE: "order" is a variable assumed available
(passed in from the outside and divided by 2 when recursing). Layout:

 - bfm_index_t block_heights[2 * order]: The height of each vertical
     block of this matrix. Even entries are the heights of T's and
     odd entries the heights of B's.
 - bfm_index_t nrows_first, nrows_second: The number of rows in
     the first and second child
 - bfm_index_t col_split: The column where we split between first
     and second child
 - char first_child[*]: The data of the first child
 - char second_child[*]: The data of the second child
 - Then follows "2 * order" interpolation matrices (pairs of (T, B)),
     each in the format specified below.

ROOT NODE: Since there is no permutation between S_maxlevel and the
blocks of D, each block of D is interleaved between the interpolation
nodes in S_maxlevel.

 - bfm_index_t block_heights[2 * order]: Sizes of final outputs (number
     of rows in each block of D)
 - bfm_index_t nrows_first, nrows_second: The number of rows in
     the first and second child
 - bfm_index_t col_split: The column where we split between first
     and second child
 - char first_child[*], second_child[*]: See abve
 - Then follows "order" instances of:
 -- bfm_index_t k_T: Number of rows in T
 -- T: Interpolation matrix
 -- Padding to 128-bit alignment
 -- D_block: Corresponding block of D in column-major format
 -- bfm_index_t k_T: Number of rows in B
 -- B: Interpolation matrix
 -- Padding to 128-bit alignment
 -- D_block: Corresponding block of D in column-major format

LEAF NODE: These are essentially "identity matrices", their sole
purpose is to instruct the parent node how many rows to consume from
the input vector. Recursion should stop in parent (since order == 1).

 - bfm_index_t n: Number of columns and rows in this identity matrix

INTERPOLATION MATRICES: Assume that number of rows "k" and number of
columns "n" is passed in from the outside. The data is:

 - char filter[n]: Exactly "k" entries will be 0, indicating the
   columns that form the identity matrix. The rest are 1.
 - Padding to 128-bit alignment
 - double data[(n - k) * k]: The data of the rest of the matrix in
   column-major order.


*/

#ifndef BUTTERFLY_H_
#define BUTTERFLY_H_

#include <stdint.h>

typedef int32_t bfm_index_t;


/*!
Multiply a butterfly matrix with a vector on the right side:

y = A * x

The previous contents of y is erased. Both \c x and \c y has
\c nvecs vectors interleaved; the matrix operates on each one.

\return 0 if success, an error code otherwise
*/
int bfm_apply_d(char *matrixdata, double *x, double *y,
                bfm_index_t nrows, bfm_index_t ncols, bfm_index_t nvecs);


#endif
