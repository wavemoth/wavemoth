/*
 *  This file is part of libc_utils.
 *
 *  libc_utils is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libc_utils is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libc_utils; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libc_utils is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*
 *  Functionality for reading wall clock time
 *
 *  Copyright (C) 2010 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#if defined (_OPENMP)
#include <omp.h>
#elif defined (USE_MPI)
#include "mpi.h"
#else
#include <sys/time.h>
#include <stdlib.h>
#endif

#include "walltime_c.h"

double wallTime(void)
  {
#if defined (_OPENMP)
  return omp_get_wtime();
#elif defined (USE_MPI)
  return MPI_Wtime();
#else
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + 1e-6*t.tv_usec;
#endif
  }

typedef struct
  {
  double ts,ta;
  int on;
  } wTimer;

#define NTIMERS 100

static wTimer wT[NTIMERS];

int wTimer_num(void)
  { return NTIMERS; }
void wTimer_reset(int n)
  { wT[n].ts=wT[n].ta=wT[n].on=0; }
void wTimer_start(int n)
  { wT[n].ts=wallTime(); wT[n].on=1; }
void wTimer_stop(int n)
  {
  if (wT[n].on)
    { wT[n].ta+=wallTime()-wT[n].ts; wT[n].on=0; }
  }
double wTimer_acc(int n)
  { return wT[n].on ? wT[n].ta+wallTime()-wT[n].ts : wT[n].ta; }
