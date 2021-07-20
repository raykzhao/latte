/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Extraction                     *
 * ****************************** */
 
#include <stdint.h>
#include "extract.h"
#include "param.h"
#include "poly.h"
#include "mat.h"
#include "fft.h"
#include "ntt.h"
#include "sample_fft.h"
#include "sample_z.h"
#include "fastrandombytes.h"
#include "red.h"

#include <mpfr.h>
#include <mpc.h>

void extract(POLY_64 *t, const MAT_64 *basis, const POLY_64 *b, const POLY_64 *a, const uint64_t l, const unsigned char *seed)
{
	static MAT_FFT fft_basis;
	static MAT_FFT g;
	static MAT_FFT tree_root;
	static POLY_FFT tree_dim2[(L + 1) * (N - 1)];
	static POLY_64 c_ntt;
	static POLY_FFT c;
	static POLY_FFT s[L + 1];
	
	mpfr_t sigma;
	mpfr_t center;
	
	mpfr_t tmp;
	
	uint64_t i, j, p;
	
	fastrandombytes_setseed(seed);
	
	mat_fft_init(&fft_basis, l + 1, N);
	mat_fft_init(&g, l + 1, N);
	
	for (i = 1; i < l + 1; i++)
	{
		for (j = 0; j < i; j++)
		{
			poly_fft_init(&(tree_root.mat[i][j]), N);
		}
	}

	for (i = 0; i < (l + 1) * (N - 1); i++)
	{
		poly_fft_init(tree_dim2 + i, N >> 1);
	}
	
	poly_fft_init(&c, N);
	
	for (i = 0; i < l + 1; i++)
	{
		poly_fft_init(s + i, N);
	}
	
	mpfr_inits2(PREC, sigma, center, tmp, NULL);
	mpfr_set_str(sigma, sigma_str[l], 10, MPFR_RNDN);
	mpfr_set_zero(center, 0);
	
	for (i = 0; i < l + 1; i++)
	{
		for (j = 0; j < l + 1; j++)
		{
			for (p = 0; p < N; p++)
			{
				mpc_set_si(fft_basis.mat[i][j].poly[p], basis->mat[i][j].poly[p], MPC_RNDNN);
			}
			
			fft(&(fft_basis.mat[i][j]), N);
		}
	}
	
	gram(&g, &fft_basis, basis, l + 1, N);
	
	fft_ldl(&tree_root, tree_dim2, &g, l + 1, sigma);
	
	/* t_{l + 1} <-- (D_{\sigma_l})^N */
	for (i = 0; i < N; i++)
	{
		t[l + 1].poly[i] = sample_z(center, sigma);
	}
	
	ntt(t + l + 1);
	
	/* c = b - t_{l + 1} * A_l */
	for (i = 0; i < N; i++)
	{
		c_ntt.poly[i] = con_add(b->poly[i] - montgomery(t[l + 1].poly[i], a->poly[i]), Q);
	}
	
	intt(&c_ntt);
	for (i = 0; i < N; i++)
	{
		mpc_set_ui(c.poly[i], c_ntt.poly[i], MPC_RNDNN);
	}
	
	fft(&c, N);
	
	/* Sample preimage */
	sample_preimage(s, &fft_basis, &tree_root, tree_dim2, &c, l + 1);
	
	for (i = 0; i < l + 1; i++)
	{
		ifft(s + i, N);
		
		for (p = 0; p < N; p++)
		{
			mpfr_round(tmp, mpc_realref(s[i].poly[p]));
			
			t[i].poly[p] = mpfr_get_si(tmp, MPFR_RNDN);
		}
		
		ntt(t + i);
	}
	
	mat_fft_clear(&fft_basis, l + 1, N);
	mat_fft_clear(&g, l + 1, N);
	
	for (i = 1; i < l + 1; i++)
	{
		for (j = 0; j < i; j++)
		{
			poly_fft_clear(&(tree_root.mat[i][j]), N);
		}
	}

	for (i = 0; i < (l + 1) * (N - 1); i++)
	{
		poly_fft_clear(tree_dim2 + i, N >> 1);
	}
	
	poly_fft_clear(&c, N);

	for (i = 0; i < l + 1; i++)
	{
		poly_fft_clear(s + i, N);
	}
	
	mpfr_clears(sigma, center, tmp, NULL);
}
