#ifndef _POLY_H
#define _POLY_H

#include "param.h"

#include <stdint.h>

#include <gmp.h>
#include <mpc.h>

#include <complex.h>
#include <math.h>

typedef struct 
{
	double complex poly[N];
} POLY_FFT;

typedef struct 
{
	mpc_t poly[N];
} POLY_FFT_HIGH;

typedef struct
{
	mpz_t poly[N];
} POLY_Z; 

typedef struct
{
	int64_t poly[N];
} POLY_64;

typedef struct
{
	double poly[N];
} POLY_R;

static inline void poly_z_init(POLY_Z *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpz_init(a->poly[i]);
	}
}

static inline void poly_z_clear(POLY_Z *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpz_clear(a->poly[i]);
	}
}

static inline void poly_fft_init_high(POLY_FFT_HIGH *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpc_init2(a->poly[i], REDUCE_K_PREC);
	}
}

static inline void poly_fft_clear_high(POLY_FFT_HIGH *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpc_clear(a->poly[i]);
	}
}

void poly_mul_zz(POLY_Z *out, const POLY_Z *a, const POLY_Z *b, const uint64_t n);

#endif
