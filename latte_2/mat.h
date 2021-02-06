/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Matrice arithmetic             *
 * ****************************** */
 
#ifndef _MAT_H
#define _MAT_H

#include "param.h"
#include "poly.h"

typedef struct 
{
	POLY_FFT mat[L + 1][L + 1];
} MAT_FFT;

typedef struct 
{
	POLY_64 mat[L + 1][L + 1];
} MAT_64;

static inline void mat_fft_init(MAT_FFT *a, const uint64_t dim, const uint64_t n)
{
	uint64_t i, j;
	
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			poly_fft_init(&(a->mat[i][j]), n);
		}
	}
}

static inline void mat_fft_clear(MAT_FFT *a, const uint64_t dim, const uint64_t n)
{
	uint64_t i, j;
	
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			poly_fft_clear(&(a->mat[i][j]), n);
		}
	}
}

void gram(MAT_FFT *out, const MAT_FFT *a, const uint64_t dim, const uint64_t n);

#endif
