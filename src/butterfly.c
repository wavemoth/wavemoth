#include "butterfly.h"
#include "stdio.h"
#include "blas.h"

#undef NDEBUG
#include "assert.h"


static INLINE char *skip_padding(char *ptr) {
  size_t m = (size_t)ptr % 16;
  if (m == 0) {
    return ptr;
  } else { 
    printf("Skipping %d bytes\n", 16 - m);
    return ptr + 16 - m;
  }
}

/*
Type implementations
*/


/*static int dense_rowmajor_right_d(char *matrixdata, double *x, double *y,
                                  bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  double *matrix = (double*)(matrixdata + 16);
  /* dgemm uses Fortran-order, so do transposed multiply;
     C-order: y^T = x^T * matrix^T
     Fortran-order: y = x * matrix
/
  int m = nvec, n = nrow, k = ncol;
  dgemm('N', 'N', m, n, k, 1.0, x, m, matrix, k, 0.0, y, m);
  return 0;
}*/


static char *filter_vectors(char *filter, double *x, double *a, double *b,
                            int32_t alen, int32_t blen, int32_t nvec) {
  int j;
  char group;
  char *end = filter + alen + blen;
  while (filter != end) {
    switch (*filter++) {
    case 0:
      for (j = 0; j != nvec; ++j) {
        *a++ = *x++;
      }
      break;
    case 1:
      for (j = 0; j != nvec; ++j) {
        *b++ = *x++;
      }
      break;
    default:
      assert(1 ? 0 : "Filter contained values besides 0 and 1");
      break;
    }
  }
  return filter;
}

static void stack_vectors_d(double *a, bfm_index_t alen, double *b, bfm_index_t blen,
                            double *output) {
  bfm_index_t i;
  for (i = 0; i != alen; ++i) {
    output[i] = a[i];
  }
  for (i = 0; i != blen; ++i) {
    output[alen + i] = b[i];
  }
}

/*
  Filter input x into two parts: Those hit with identity matrix which goes
  to y, and those going to a set of temporary vectors tmp_vecs which
  will be multiplied with the interpolation matrix and then added to y.
  
  input is n-by-nvec, output is k-by-nvec
*/
static char *apply_interpolation_d(char *head, double *input, double *output,
                                   int32_t k, int32_t n, int32_t nvec) {
  int i;
  double tmp_vecs[nvec * (n - k)];
  head = filter_vectors(head, input, output, tmp_vecs, k, n - k, nvec);
  head = skip_padding(head);
  printf("-- %d %d\n", k, n);
  dgemm_crr((double*)head, tmp_vecs, output, k, nvec, n - k, 1.0);
  head += sizeof(double[k * (n - k)]);
  return head;
}

/* Forward declaration */
static char *apply_butterfly_node_d(char *head, bfm_index_t order, double *input,
                                    double *output, double *buffer,
                                    bfm_index_t nrows, bfm_index_t ncols,
                                    bfm_index_t nvecs);

static INLINE char *recurse_d(char *head, bfm_index_t order,
                              double *input, bfm_index_t ncols,
                              bfm_index_t nvecs,
                              double *output,
                              double *buffer2,
                              double **data_from_first,
                              double **data_from_second,
                              bfm_index_t **out_block_heights,
                              bfm_index_t **out_block_widths_first,
                              bfm_index_t **out_block_widths_second) {
  bfm_index_t *block_widths_first, *block_widths_second;
  *out_block_heights = (bfm_index_t*)head; /* [2 * order] */
  bfm_index_t nrows_first = ((bfm_index_t*)head)[2 * order];
  bfm_index_t nrows_second = ((bfm_index_t*)head)[2 * order + 1];
  bfm_index_t col_split = ((bfm_index_t*)head)[2 * order + 2];
  head += sizeof(bfm_index_t[2 * order + 3]);
  printf("RECURSE %d \n", order);
  if (order == 1) {
    /* Parse the two leaf node identity matrices */
    *out_block_widths_first = (bfm_index_t*) head;
    *out_block_widths_second = ((bfm_index_t*) head) + 1;
    head += sizeof(bfm_index_t[2]);
    nrows_first = (*out_block_widths_first)[0];
    *data_from_first = input;
    *data_from_second = *data_from_first + nrows_first * nvecs;
  } else {
    printf("BODY\n");
    *data_from_first = output;
    /* Recurse to sub-nodes. */
    *out_block_widths_first = (bfm_index_t*) head;
    head = apply_butterfly_node_d(head, order / 2, input, *data_from_first, buffer2, nrows_first,
                                  col_split, nvecs);
    printf("MID\n");
    input += col_split * nvecs;
    *data_from_second = *data_from_first + nrows_first * nvecs;
    *out_block_widths_second = (bfm_index_t*) head;
    head = apply_butterfly_node_d(head, order / 2, input, *data_from_second, buffer2, nrows_second,
                                  ncols - col_split, nvecs);
  }
  printf("DONE RECURSE %d \n", order);
  return head;
}

static char *apply_butterfly_node_d(char *head, bfm_index_t order, double *input,
                                    double *output, double *buffer,
                                    bfm_index_t nrows, bfm_index_t ncols,
                                    bfm_index_t nvecs) {
  bfm_index_t *block_heights, *block_widths_first, *block_widths_second;
  bfm_index_t i;
  double *data_from_first, *data_from_second;

  head = recurse_d(head, order, input, ncols, nvecs, buffer, output,
                   &data_from_first, &data_from_second, 
                   &block_heights, &block_widths_first, &block_widths_second);

  /* Apply interpolation matrices */
  for (i = 0; i != order; ++i) {
    bfm_index_t n = block_widths_first[i] + block_widths_second[i];
    double in_buf[n * nvecs];
    /* Merge input vectors for this column */
    stack_vectors_d(data_from_first, block_widths_first[i] * nvecs,
                    data_from_second, block_widths_second[i] * nvecs,
                    in_buf);
    data_from_first += block_widths_first[i] * nvecs;
    data_from_second += block_widths_second[i] * nvecs;
      
    /* T_ip and T_k */
    head = apply_interpolation_d(head, data_from_first, output, block_heights[2 * i], n, nvecs);
    output += block_heights[2 * i] * nvecs;
    /* B_ip and B_k */
    head = apply_interpolation_d(head, data_from_first, output, block_heights[2 * i + 1], n, nvecs);
    output += block_heights[2 * i + 1] * nvecs;        
  }
  return head;
}

static INLINE char *apply_root_block_d(char *head,
                                       double *input, double *output,
                                       bfm_index_t m, bfm_index_t n,
                                       bfm_index_t nvecs) {
  bfm_index_t k = ((bfm_index_t*)head)[0];
  double buf[k * nvecs];
  head += sizeof(bfm_index_t);
  head = apply_interpolation_d(head, input, buf, k, n, nvecs);
  head = skip_padding(head);
  printf("> %f %d %d %d\n", buf[0], m, nvecs, k);
  dgemm_crr((double*)head, buf, output, m, nvecs, k, 0.0);
  printf("++ %f %f\n", output[0], output[1]);
  head += sizeof(double[m * k]);
  return head;
}

int bfm_apply_d(char *head, double *x, double *y,
                bfm_index_t nrows, bfm_index_t ncols, bfm_index_t nvecs) {
  char *data = head;
  bfm_index_t order, *block_heights, *block_widths_first, *block_widths_second;
  double *data_from_first, *data_from_second;
  bfm_index_t i;
  double buffer[nrows * nvecs * 1000], buffer2[nrows * nvecs * 1000]; /* TODO: Make sure this is sufficient. */
  assert((size_t)head % 16 == 0); /* Ensure data alignment */
  order = ((bfm_index_t*)head)[0];
  assert(order >= 1);
  head += sizeof(bfm_index_t);
  head = recurse_d(head, order, x, ncols, nvecs, buffer, buffer2,
                   &data_from_first, &data_from_second, 
                   &block_heights, &block_widths_first, &block_widths_second);

  /* Now, apply interpolation matrices as well as final dense diagonal blocks */
  for (i = 0; i != order; ++i) {
    bfm_index_t n = block_widths_first[i] + block_widths_second[i];
    double in_buf[n * nvecs];
    /* Merge input vectors for this column */
    stack_vectors_d(data_from_first, block_widths_first[i] * nvecs,
                    data_from_second, block_widths_second[i] * nvecs,
                    in_buf);
    data_from_first += block_widths_first[i] * nvecs;
    data_from_second += block_widths_second[i] * nvecs;
      
    /* T_ip and T_k */
    head = apply_root_block_d(head, in_buf, y, block_heights[2 * i], n, nvecs);
    y += block_heights[2 * i] * nvecs;
    nrows -= block_heights[2 * i] * nvecs;
    printf("checkpoint %ld %d\n", (size_t)(head - data), nrows);
    /* B_ip and B_k */
    head = apply_root_block_d(head, in_buf, y, block_heights[2 * i + 1], n, nvecs);
    y += block_heights[2 * i + 1] * nvecs;        
    nrows -= block_heights[2 * i + 1] * nvecs;
    printf("checkpoint %ld %d\n", (size_t)(head - data), nrows);
  }
  printf("DONE\n");
  return 0;
}

  /*
static char *apply_interpolation_d(char *head, double *input, double *output,
                                   int32_t k, int32_t n, int32_t nvec) {
static INLINE void dgemm_crr(double *A, double *X, double *Y,
                              int32_t m, int32_t n, int32_t k, beta) {
  */
