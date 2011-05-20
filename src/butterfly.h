/*!
Code for applying a butteryfly-compressed matrix to a vector.

TODO: Also contains hard-coded support for Associated Legendre
recursion.


\section Binary format

A compressed matrix (or matrix sub-block) is represented as a) a
32-bit type field which designates the matrix type, and then
block-specific data. Contents should typically be 128-bit aligned, so
each block should start with 96 bits/12 bytes of padding or
block-type-specific header fields. The block size should be divisible
by 16 bytes/128 bits as well.

Some matrix types are recursive, but that's defined by the block type.

Matrix dimensions are always passed to code from the caller and
is not part of the matrix data.

\c ZERO: Fills the output vector with zeros. Only contains 12 bytes of
padding.

\c DENSE_ROWMAJOR: 12 bytes of padding, then raw double precision matrix
data in row-major ordering.

\c HSTACK: Matrix blocked horizontally. First an int32 \c n with the
number of blocks, then comes an \c n-length array of type
BFM_StackedBlock with the column widths and offsets w.r.t the
beginning of the block (including type id) of corresponding matrix
data blocks.

\c BUTTERFLY: Consists of 4 dense interpolation matrices + associated
permuted identity matrices, dubbed L_p, R_p, T_p, B_p, as well as
two sub-matrices of generic type, T_k and B_k. Storage format:

 - BFM_ButterflyHeader
 - LR_split * 1 byte: Filter: 1 if column is part of identity matrix, 0 otherwise
 - Padding to next 128-bit aligned address using the char 0xFF.
 - Contents of L_p in col-major order
 - (ncol - LR_split) * 1 byte + padding: Filter for R
 - Contents of R_p in row-major order
 - T_p: First filter, padding, T_p in row-major order, padding to next 128 bit.
 - T_k: Generic sub-matrix
 - Padding
 - B: Filter, padding, B_p, padding, B_k

*/

#ifndef BUTTERFLY_H_
#define BUTTERFLY_H_

#include <stdint.h>

typedef int32_t bfm_index_t;

typedef enum {
  BFM_BLOCK_ZERO = 0,
  BFM_BLOCK_DENSE_ROWMAJOR = 1,
  BFM_BLOCK_HSTACK = 2,
  BFM_BLOCK_BUTTERFLY = 3,
  BFM_MAX_TYPE = 3
} BFM_MatrixBlockType;

/*
 - int32: ncol(L_p), this also yields ncol(R_p) through ncol([L_p R_p]) - ncol(L_P)
 - int32: nrow(L_p)
 - int32: nrow(R_p)
 - int32: max(ncol(L_p), ncol(R_p)): Useful for memory allocation purposes

ncol(T_p) = nrow(L_p) + nrow(R_p)
*/
typedef struct {
  int32_t type_id;
  int32_t ncol_L_p, nrow_L_p, nrow_R_p, nrow_buf;
} BFM_ButterflyHeader;




/*typedef struct {
  bfm_index_t num_cols_or_rows;
  size_t data_offset;
} BFM_StackedBlock;

typedef struct {
} BFM_Context;*/

/*!
Multiply a compressed matrix with a vector on the right side.
The matrix is assumed to be real and is applied to both the real
and imaginary parts of the input vectors (_dz).

\c x is the input vector, and \c y is the output vector. Each has
the data for \c nvecs vectors interleaved; the matrix operates
on each one.

\return 0 if success, an error code otherwise
*/
int bfm_apply_right_d(char *matrixdata, double *x, double *y,
                      bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec);


#endif
