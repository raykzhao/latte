/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Polynomial arithmetic          *
 * ****************************** */
 
#ifndef _POLY_H
#define _POLY_H

#include "param.h"

#include <stdint.h>

#include <gmp.h>
#include <mpc.h>

typedef struct 
{
	mpc_t poly[N];
} POLY_FFT;

typedef struct
{
	mpz_t poly[N];
} POLY_Z; 

typedef struct
{
	int64_t poly[N];
} POLY_64;

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

static inline void poly_fft_init(POLY_FFT *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpc_init2(a->poly[i], PREC);
	}
}

static inline void poly_fft_clear(POLY_FFT *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpc_clear(a->poly[i]);
	}
}

void poly_mul_zz(POLY_Z *out, const POLY_Z *a, const POLY_Z *b, const uint64_t n);
void poly_mul_z64(POLY_Z *out, const POLY_Z *a, const POLY_64 *b, const uint64_t n);

#endif
