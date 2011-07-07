#ifndef _LEGENDRETRANSFORM_H_
#define _LEGENDRETRANSFORM_H_

#include <stddef.h>

/*
Code to do an associated Legendre function transform (or its
transpose) given some starting values; for the even/odd part
seperately. Uses a three-term recurrence relation that only involves
even or odd l's.
*/

void fastsht_associated_legendre_transform(size_t nx, size_t nl,
                                           size_t nvecs,
                                           size_t *il_start, 
                                           double *a_l,
                                           double *y,
                                           double *x_squared, 
                                           double *c, double *d,
                                           double *c_inv,
                                           double *P, double *Pp1);

void fastsht_associated_legendre_transform_sse(size_t nx, size_t nl,
                                               size_t nvecs,
                                               size_t *il_start, 
                                               double *a_l,
                                               double *y,
                                               double *x_squared, 
                                               double *c, double *d,
                                               double *c_inv,
                                               double *P, double *Pp1);

#endif
