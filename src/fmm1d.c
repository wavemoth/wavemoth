#include "fmm1d.h"
#include "compat.h"
#include "math.h"
#include "malloc.h"
#include "assert.h"
#include "unistd.h"
#include "fastsht_error.h"

#define NQUAD 28

static double QUAD_POINTS[28] SSE_ALIGNED;
static double QUAD_WEIGHTS[28] SSE_ALIGNED;

/* struct _fastsht_fmm1d { */
/*   double *dx; */

/* }; */

/* fastsht_fmm1d* fastsht_initialize_fmm1d(double *in_grid, double *out_grid, */
/*                                         size_t N_in, size_t N_out) { */
/*   fastsht_fmm1d *ctx = malloc(sizeof(fastsht_fmm1d)); */
/*   size_t i; */
/*   for (i = 0; i != N_in; ++i) { */
/*   } */
/*   return ctx; */
/* } */
                             
/* void fastsht_destruct_fmm1d(fastsht_fmm1d *ctx) { */
/*   free(info); */
/* } */

void fastsht_fmm1d(const double *restrict x_grid, const double *restrict input_x, size_t nx,
                   const double *restrict y_grid, double *restrict output_y, size_t ny,
                   size_t nvecs) {
  /*
    x_far - The position of the far-field expansion
    ix_far - Index (in x_grid) for far-field expansion, positioned one
     element past the last grid point of the far field
   */
  double max_dist, s, r, x_far, dx, y, exp_term;
  ssize_t i, j, k, ix, iy, ix_far;

  double *restrict quad_weights, *restrict quad_points, *restrict alpha,
         *restrict buf;
  /* Corner cases */
  if (ny == 0 || nvecs == 0) return;
  if (nx == 0) {
    for (i = 0; i != ny * nvecs; ++i) output_y[i] = 0;
    return;
  }

  /* Allocate aligned buffers for quadrature points, quadrature
     weights, and expansion. */
  assert((NQUAD * sizeof(double)) % 16 == 0);
  buf = memalign(16, sizeof(double[(3 + nvecs) * NQUAD]));
  quad_weights  = buf + NQUAD;
  quad_points = buf + 2 * NQUAD;
  alpha = buf + 3 * NQUAD; /* size nvecs * NQUAD */

  /* Find maximum distance between grid points in order to rescale
     quadrature to safe range. */
  max_dist = fmax(x_grid[nx - 1], y_grid[ny - 1]) - fmin(x_grid[0], y_grid[0]);
  s = 500. / max_dist; /* Rescale factor */
  r = 1. / s; /* Range of near-field */
  /* Initialize rescaled quadrature weights and expansion */
  for (k = 0; k != NQUAD; ++k) {
    quad_weights[k] = QUAD_WEIGHTS[k] * s;
    quad_points[k] = QUAD_POINTS[k] * s;
    for (j = 0; j != nvecs; ++j) {
      alpha[k * nvecs + j] = 0.;
    }
  }

  /* Do rightwards pass. */
  ix_far = 0;
  /* Note: The initial value for x_far does not matter in principle, because
     alpha[k] == 0. However we at least want to avoid exp(...) to over/underflow.
     So we set x_far so that dx == 0 in the first iteration.
  */
  x_far = fmin(x_grid[0], y_grid[0]);
  for (iy = 0; iy != ny; ++iy) {
    y = y_grid[iy];
    /* Translate and update far-field expansion of input values until
       it is within r of our output evaluation point. */
    while (ix_far < nx && x_grid[ix_far] < y - r) {
      dx = x_far - x_grid[ix_far];
      x_far = x_grid[ix_far];
      for (k = 0; k != NQUAD; ++k) {
        exp_term = exp(dx * quad_points[k]);
        for (j = 0; j != nvecs; ++j) {
          alpha[k * nvecs + j] = alpha[k * nvecs + j] * exp_term + input_x[ix_far * nvecs + j];
        }
      }
      ++ix_far;
    }
    /* Evaluate far-field expansion at output point. */
    dx = x_far - y;
    for (j = 0; j != nvecs; ++j) output_y[iy * nvecs + j] = 0;
    for (k = 0; k != NQUAD; ++k) {
      exp_term = quad_weights[k] * exp(dx * quad_points[k]);
      for (j = 0; j != nvecs; ++j) {
        output_y[iy * nvecs + j] += alpha[k * nvecs + j] * exp_term;
      }
    }
    /* Brute-force computation of near-field contribution (both
       to left and right of evaluation point, while we're at it). */
    for (ix = ix_far; ix < nx && x_grid[ix] <= y + r; ++ix) {
      /* This issue must be studied further if it arises */
      checkf(fabs(y - x_grid[ix]) / max_dist > 1e-7,
             "Evaluation point %e too close to grid point %e relative to max distance %e",
             y, x_grid[ix], max_dist);
      exp_term = 1 / (y - x_grid[ix]);
      for (j = 0; j != nvecs; ++j) {
        output_y[iy * nvecs + j] += input_x[ix * nvecs + j] * exp_term;
      }
    }
  }
  /* Leftwards pass for far-field. This is the same, but now we rescale
     the quadrature by -1, which changes some signs. Note the use
     of ssize_t to make loop logic more readable. */
  for (k = 0; k != NQUAD; ++k) {
    for (j = 0; j != nvecs; ++j) alpha[k * nvecs + j] = 0;
  }
  ix_far = nx - 1;
  x_far = fmax(x_grid[nx - 1], y_grid[ny - 1]);
  for (iy = ny - 1; iy != -1; iy--) {
    y = y_grid[iy];
    /* Translate & update right far-field expansion*/
    while (ix_far > -1 && x_grid[ix_far] > y + r) {
      dx = x_grid[ix_far] - x_far;
      x_far = x_grid[ix_far];
      for (k = 0; k != NQUAD; ++k) {
        exp_term = exp(dx * quad_points[k]);
        for (j = 0; j != nvecs; ++j) {
          alpha[k * nvecs + j] = alpha[k * nvecs + j] * exp_term + input_x[ix_far * nvecs + j];
        }
      }
      --ix_far;
    }
    /* Evaluate expansion and add contribution */
    dx = y - x_far;
    for (k = 0; k != NQUAD; ++k) {
      exp_term = quad_weights[k] * exp(dx * quad_points[k]);
      for (j = 0; j != nvecs; ++j) {
        output_y[iy * nvecs + j] -= alpha[k * nvecs + j] * exp_term;
      }
    }

  }
  free(buf); /* Other temps are in same buffer! */
}



/*
Weights for generalized Gaussian quadrature of

   int_0^\infty e^{tx} dx = sum w_i exp(t * x_i)

with t in the range [1, 501].

Taken from http://www.netlib.org/pdes/multipole/wts500.f

N. Yarvin, V. Rokhlin, Generalized Gaussian Quadratures and Singular
Value Decompositions of Integral Operators, 1998,
SIAM J. Sci. Comput. 20, pp. 699-718 
 */


static double QUAD_POINTS[28] SSE_ALIGNED = {
 0.51381795837439443363513212680713947e-03,
 0.27304457410704199858650659828620367e-02,
 0.68160137769119765426643490207146340e-02,
 0.12952008943105054557760398423212956e-01,
 0.21430556623995591107026232001066913e-01,
 0.32693077435416129189765100617149847e-01,
 0.47384339361452651219408943088094832e-01,
 0.66422538658869906464895332297645726e-01,
 0.91083609265701992171879031038032271e-01,
 0.12309931515680468843361103443044334e+00,
 0.16477454361350713713658223760718724e+00,
 0.21913585560877712250760133646822826e+00,
 0.29012664768975497883815066595179815e+00,
 0.38286487674840579443374001038670456e+00,
 0.50398058179729621330814443966676445e+00,
 0.66205451410726746024107068348273274e+00,
 0.86818664253955900521872813145997494e+00,
 0.11367350670636899400604869200366967e+01,
 0.14862849200104679887431662518488448e+01,
 0.19409402276831122951996033127610769e+01,
 0.25320952507423009788362491203678215e+01,
 0.33009718613493033820591483341014688e+01,
 0.43024948141012255149771188379035923e+01,
 0.56117545045133971321428621533897157e+01,
 0.73360976078470116280524716610880778e+01,
 0.96414187592042593009284671180300365e+01,
 0.12823096392169689557014539066503465e+02,
 0.17587400071042305461725873529323834e+02
};

static double QUAD_WEIGHTS[28] SSE_ALIGNED = {
 0.13212378315760757100801294124815478e-02,
 0.31285477801160114283501913408130908e-02,
 0.50723762615025144279692888527822532e-02,
 0.72476698125814896247817149986171532e-02,
 0.97820955598317266921336611328577874e-02,
 0.12849233584715652412922486858932270e-01,
 0.16684914458309663575212478712400069e-01,
 0.21602517467719514930691067702478109e-01,
 0.28006326728639319667000527469785398e-01,
 0.36406794940307839233444412699072219e-01,
 0.47444873633013210207704765440613480e-01,
 0.61930727493054842346818886062879310e-01,
 0.80898273560459582900261577117872215e-01,
 0.10567594748292560400210313567244803e+00,
 0.13797620661979101239766714101470269e+00,
 0.18000947429659496345933986602370689e+00,
 0.23463186153608726981379368239281040e+00,
 0.30554132794139847744947603213040709e+00,
 0.39754682244155206328794798554812257e+00,
 0.51695511158335980560554454196568031e+00,
 0.67216328370835447637864341433166332e+00,
 0.87464250694849374310652866430377750e+00,
 0.11407332853853718344895028360165344e+01,
 0.14952907674277354153396565164371127e+01,
 0.19800794802566744638917545875047599e+01,
 0.26766284602650483548086237399782999e+01,
 0.37870913379223214170502998382833988e+01,
 0.61141264493435815963177300026057897e+01
};
