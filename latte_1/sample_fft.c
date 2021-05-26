/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Lattice Gaussian sampler       *
 * ****************************** */
 
#include <stdint.h>
#include "sample_fft.h"
#include "param.h"
#include "poly.h"
#include "mat.h"
#include "fft.h"
#include "sample_z.h"

#include <mpfr.h>
#include <mpc.h>

/* LDL from Falcon (for dim = 2) */
static void ldl_dim2(POLY_FFT *l_dim2, POLY_FFT *d, const MAT_FFT *g, const uint64_t n)
{
	uint64_t p;
	mpc_t tmp;
	
	mpc_init2(tmp, PREC);
	
	for (p = 0; p < n; p++)
	{
		mpc_set(d[0].poly[p], g->mat[0][0].poly[p], MPC_RNDNN);
		mpc_set(d[1].poly[p], g->mat[1][1].poly[p], MPC_RNDNN);

		mpc_div(l_dim2->poly[p], g->mat[1][0].poly[p], d[0].poly[p], MPC_RNDNN);

		mpc_conj(tmp, l_dim2->poly[p], MPC_RNDNN);
		mpc_mul(tmp, g->mat[1][0].poly[p], tmp, MPC_RNDNN);
		mpc_sub(d[1].poly[p], d[1].poly[p], tmp, MPC_RNDNN);
	}
	
	mpc_clear(tmp);
}

/* LDL from Falcon */
static void ldl(MAT_FFT *l, POLY_FFT *d, const MAT_FFT *g, const uint64_t dim, const uint64_t n)
{
	uint64_t i, j, k, p;
	mpc_t tmp1, tmp2;
	
	mpc_init2(tmp1, PREC);
	mpc_init2(tmp2, PREC);
	
	for (i = 0; i < dim; i++)
	{
		for (p = 0; p < n; p++)
		{
			mpc_set(d[i].poly[p], g->mat[i][i].poly[p], MPC_RNDNN);
		}
		
		for (j = 0; j < i; j++)
		{
			for (p = 0; p < n; p++)
			{
				mpc_set(l->mat[i][j].poly[p], g->mat[i][j].poly[p], MPC_RNDNN);
			}
			
			for (k = 0; k < j; k++)
			{
				for (p = 0; p < n; p++)
				{
					mpc_conj(tmp1, l->mat[j][k].poly[p], MPC_RNDNN);
					mpc_mul(tmp2, l->mat[i][k].poly[p], tmp1, MPC_RNDNN);
					mpc_mul(tmp2, tmp2, d[k].poly[p], MPC_RNDNN);
					mpc_sub(l->mat[i][j].poly[p], l->mat[i][j].poly[p], tmp2, MPC_RNDNN);
				}
			}
			
			for (p = 0; p < n; p++)
			{
				mpc_set(tmp2, l->mat[i][j].poly[p], MPC_RNDNN);
				mpc_div(l->mat[i][j].poly[p], l->mat[i][j].poly[p], d[j].poly[p], MPC_RNDNN);
				
				mpc_conj(tmp1, l->mat[i][j].poly[p], MPC_RNDNN);
				mpc_mul(tmp2, tmp2, tmp1, MPC_RNDNN);
				mpc_sub(d[i].poly[p], d[i].poly[p], tmp2, MPC_RNDNN);
			}
		}
	}
	
	mpc_clear(tmp1);
	mpc_clear(tmp2);
}

/* ffLDL from Falcon (for dim = 2) */
static void fft_ldl_dim2(POLY_FFT *tree_dim2, const MAT_FFT *g, const uint64_t n, const mpfr_t sigma)
{
	uint64_t p;
	uint64_t n2 = n >> 1;
	
	static MAT_FFT g_i;
	static POLY_FFT d0, d1;
	static uint64_t initialised; 

	POLY_FFT d[2];
	
	mpfr_t tmp;
	
	if (n == 1)
	{
		mpfr_init2(tmp, PREC);

		mpfr_sqrt(tmp, mpc_realref(g->mat[0][0].poly[0]), MPFR_RNDN);
		mpfr_div(tmp, sigma, tmp, MPFR_RNDN);
		mpc_set_fr(tree_dim2->poly[0], tmp, MPC_RNDNN);

		mpfr_clear(tmp);
	}
	else
	{
		if (!initialised)
		{
			mat_fft_init(&g_i, 2, N >> 2);
					
			poly_fft_init(&d0, N >> 2);
			poly_fft_init(&d1, N >> 2);
			
			initialised = 1;
		}
		
		poly_fft_init(d, n);
		poly_fft_init(d + 1, n);
		
		ldl_dim2(tree_dim2, d, g, n);
		
		split_fft(&d0, &d1, d, n);
		
		for (p = 0; p < n2; p++)
		{
			mpc_set(g_i.mat[0][0].poly[p], d0.poly[p], MPC_RNDNN);
			mpc_set(g_i.mat[0][1].poly[p], d1.poly[p], MPC_RNDNN);
			mpc_conj(g_i.mat[1][0].poly[p], d1.poly[p], MPC_RNDNN);
			mpc_set(g_i.mat[1][1].poly[p], d0.poly[p], MPC_RNDNN);
		}
		
		fft_ldl_dim2(tree_dim2 + 1, &g_i, n2, sigma);
		
		split_fft(&d0, &d1, d + 1, n);
		
		for (p = 0; p < n2; p++)
		{
			mpc_set(g_i.mat[0][0].poly[p], d0.poly[p], MPC_RNDNN);
			mpc_set(g_i.mat[0][1].poly[p], d1.poly[p], MPC_RNDNN);
			mpc_conj(g_i.mat[1][0].poly[p], d1.poly[p], MPC_RNDNN);
			mpc_set(g_i.mat[1][1].poly[p], d0.poly[p], MPC_RNDNN);
		}
		
		fft_ldl_dim2(tree_dim2 + n, &g_i, n2, sigma);
		
		poly_fft_clear(d, n);
		poly_fft_clear(d + 1, n);
		
		if (n == N >> 1)
		{
			mat_fft_clear(&g_i, 2, N >> 2);
					
			poly_fft_clear(&d0, N >> 2);
			poly_fft_clear(&d1, N >> 2);
			
			initialised = 0;
		}
	}
}

/* ffLDL from Falcon 
 * since all nodes except the root only have L[1, 0], we separate the root from the remaining of the tree to save space */
void fft_ldl(MAT_FFT *tree_root, POLY_FFT *tree_dim2, const MAT_FFT *g, const uint64_t dim, const mpfr_t sigma)
{
	uint64_t i, p;
	
	static MAT_FFT g_i;
	static POLY_FFT d0, d1;
	
	static POLY_FFT d[L + 1];
	
	mat_fft_init(&g_i, 2, N >> 1);
			
	poly_fft_init(&d0, N >> 1);
	poly_fft_init(&d1, N >> 1);
	
	for (i = 0; i < dim; i++)
	{
		poly_fft_init(d + i, N);
	}
	
	ldl(tree_root, d, g, dim, N);
	
	for (i = 0; i < dim; i++)
	{
		split_fft(&d0, &d1, d + i, N);
		
		for (p = 0; p < N >> 1; p++)
		{
			mpc_set(g_i.mat[0][0].poly[p], d0.poly[p], MPC_RNDNN);
			mpc_set(g_i.mat[0][1].poly[p], d1.poly[p], MPC_RNDNN);
			mpc_conj(g_i.mat[1][0].poly[p], d1.poly[p], MPC_RNDNN);
			mpc_set(g_i.mat[1][1].poly[p], d0.poly[p], MPC_RNDNN);
		}
		
		fft_ldl_dim2(tree_dim2 + i * (N - 1), &g_i, N >> 1, sigma);
	}
		
	for (i = 0; i < dim; i++)
	{
		poly_fft_clear(d + i, N);
	}
	
	mat_fft_clear(&g_i, 2, N >> 1);
	
	poly_fft_clear(&d0, N >> 1);
	poly_fft_clear(&d1, N >> 1);
}

/* ffSampling from Falcon (for dim = 2) */
static void fft_sampling_dim2(POLY_FFT *z, const POLY_FFT *tree_dim2, const POLY_FFT *t, const uint64_t n)
{
	uint64_t n2 = n >> 1;
	
	uint64_t p;
	
	static POLY_FFT t_j;
	static uint64_t initialised; 

	mpc_t tmp;
	
	POLY_FFT t_j_split[2];
	POLY_FFT z_j_split[2];
	
	if (n == 1)
	{
		mpc_set_si(z[0].poly[0], sample_z(mpc_realref(t[0].poly[0]), mpc_realref(tree_dim2->poly[0])), MPC_RNDNN);
		mpc_set_si(z[1].poly[0], sample_z(mpc_realref(t[1].poly[0]), mpc_realref(tree_dim2->poly[0])), MPC_RNDNN);
	}
	else
	{
		if (!initialised)
		{
			poly_fft_init(&t_j, N >> 1);
			
			initialised = 1;
		}

		mpc_init2(tmp, PREC);
		
		poly_fft_init(t_j_split, n2);
		poly_fft_init(t_j_split + 1, n2);
		
		poly_fft_init(z_j_split, n2);
		poly_fft_init(z_j_split + 1, n2);
		
		for (p = 0; p < n; p++)
		{
			mpc_set(t_j.poly[p], t[1].poly[p], MPC_RNDNN);
		}
		
		split_fft(t_j_split, t_j_split + 1, &t_j, n);
		
		fft_sampling_dim2(z_j_split, tree_dim2 + n, t_j_split, n2);
		
		merge_fft(z + 1, z_j_split, z_j_split + 1, n);
		
		for (p = 0; p < n; p++)
		{
			mpc_set(t_j.poly[p], t[0].poly[p], MPC_RNDNN);
			
			mpc_sub(tmp, t[1].poly[p], z[1].poly[p], MPC_RNDNN);
			mpc_fma(t_j.poly[p], tmp, tree_dim2->poly[p], t_j.poly[p], MPC_RNDNN);
		}
		
		split_fft(t_j_split, t_j_split + 1, &t_j, n);
		
		fft_sampling_dim2(z_j_split, tree_dim2 + 1, t_j_split, n2);
		
		merge_fft(z, z_j_split, z_j_split + 1, n);
		
		mpc_clear(tmp);
		
		poly_fft_clear(t_j_split, n2);
		poly_fft_clear(t_j_split + 1, n2);
		
		poly_fft_clear(z_j_split, n2);
		poly_fft_clear(z_j_split + 1, n2);

		if (n == N >> 1)
		{
			poly_fft_clear(&t_j, N >> 1);
			
			initialised = 0;
		}
	}
}

/* ffSampling from Falcon */
static void fft_sampling(POLY_FFT *z, const MAT_FFT *tree_root, const POLY_FFT *tree_dim2, const POLY_FFT *t, const uint64_t dim)
{
	int64_t i, j, p;
	
	static POLY_FFT t_j;
	
	mpc_t tmp;
	
	static POLY_FFT t_j_split[2];
	static POLY_FFT z_j_split[2];
	
	poly_fft_init(&t_j, N);
	
	mpc_init2(tmp, PREC);
	
	poly_fft_init(t_j_split, N >> 1);
	poly_fft_init(t_j_split + 1, N >> 1);
	
	poly_fft_init(z_j_split, N >> 1);
	poly_fft_init(z_j_split + 1, N >> 1);
	
	for (j = dim - 1; j >= 0; j--)
	{
		for (p = 0; p < N; p++)
		{
			mpc_set(t_j.poly[p], t[j].poly[p], MPC_RNDNN);
		}
		
		for (i = j + 1; i < dim; i++)
		{
			for (p = 0; p < N; p++)
			{
				mpc_sub(tmp, t[i].poly[p], z[i].poly[p], MPC_RNDNN);
				mpc_fma(t_j.poly[p], tmp, tree_root->mat[i][j].poly[p], t_j.poly[p], MPC_RNDNN);
			}
		}
		
		split_fft(t_j_split, t_j_split + 1, &t_j, N);
		
		fft_sampling_dim2(z_j_split, tree_dim2 + j * (N - 1), t_j_split, N >> 1);
		
		merge_fft(z + j, z_j_split, z_j_split + 1, N);
	}
	
	mpc_clear(tmp);
	
	poly_fft_clear(t_j_split, N >> 1);
	poly_fft_clear(t_j_split + 1, N >> 1);
	
	poly_fft_clear(z_j_split, N >> 1);
	poly_fft_clear(z_j_split + 1, N >> 1);
	
	poly_fft_clear(&t_j, N);
}

/* Sample preimage 
 * Equivalent to (c, 0) - GPV(B, \sigma, c) */
void sample_preimage(POLY_FFT *s, const MAT_FFT *b, const MAT_FFT *tree_root, const POLY_FFT *tree_dim2, const POLY_FFT *c, const uint64_t dim)
{
	static POLY_FFT t[L + 1];
	static POLY_FFT z[L + 1];
	
	mpc_t tmp1, tmp2;
	
	uint64_t i, j, p;
	
	for (i = 0; i < dim; i++)
	{
		poly_fft_init(t + i, N);
		poly_fft_init(z + i, N);
	}
	
	mpc_init2(tmp1, PREC);
	mpc_init2(tmp2, PREC);
	
	/* t = (c, 0) * B^{-1}
	 * Here we only consider dim = 2 or 3 */
	if (dim == 2)
	{
		for (p = 0; p < N; p++)
		{
			mpc_mul(t[0].poly[p], c->poly[p], b->mat[1][1].poly[p], MPC_RNDNN);
			mpc_div_ui(t[0].poly[p], t[0].poly[p], Q, MPC_RNDNN);
			
			mpc_neg(tmp1, b->mat[0][1].poly[p], MPC_RNDNN);
			mpc_mul(t[1].poly[p], c->poly[p], tmp1, MPC_RNDNN);
			mpc_div_ui(t[1].poly[p], t[1].poly[p], Q, MPC_RNDNN);
		}
	}
	else if (dim == 3)
	{
		for (p = 0; p < N; p++)
		{
			mpc_mul(tmp1, b->mat[1][1].poly[p], b->mat[2][2].poly[p], MPC_RNDNN);
			mpc_mul(tmp2, b->mat[1][2].poly[p], b->mat[2][1].poly[p], MPC_RNDNN);
			mpc_sub(tmp1, tmp1, tmp2, MPC_RNDNN);
			mpc_mul(t[0].poly[p], c->poly[p], tmp1, MPC_RNDNN);
			mpc_div_ui(t[0].poly[p], t[0].poly[p], Q, MPC_RNDNN);
			
			mpc_mul(tmp1, b->mat[0][2].poly[p], b->mat[2][1].poly[p], MPC_RNDNN);
			mpc_mul(tmp2, b->mat[0][1].poly[p], b->mat[2][2].poly[p], MPC_RNDNN);
			mpc_sub(tmp1, tmp1, tmp2, MPC_RNDNN);
			mpc_mul(t[1].poly[p], c->poly[p], tmp1, MPC_RNDNN);
			mpc_div_ui(t[1].poly[p], t[1].poly[p], Q, MPC_RNDNN);		

			mpc_mul(tmp1, b->mat[0][1].poly[p], b->mat[1][2].poly[p], MPC_RNDNN);
			mpc_mul(tmp2, b->mat[0][2].poly[p], b->mat[1][1].poly[p], MPC_RNDNN);
			mpc_sub(tmp1, tmp1, tmp2, MPC_RNDNN);
			mpc_mul(t[2].poly[p], c->poly[p], tmp1, MPC_RNDNN);
			mpc_div_ui(t[2].poly[p], t[2].poly[p], Q, MPC_RNDNN);		
		}
	}
	
	/* z = ffSampling(t, tree) */
	fft_sampling(z, tree_root, tree_dim2, t, dim);
	
	/* preimage s = (t - z) * B */
	for (i = 0; i < dim; i++)
	{
		for (p = 0; p < N; p++)
		{
			mpc_sub(z[i].poly[p], t[i].poly[p], z[i].poly[p], MPC_RNDNN);
		}
	}
	
	for (i = 0; i < dim; i++)
	{
		for (p = 0; p < N; p++)
		{
			mpc_set_ui(s[i].poly[p], 0, MPC_RNDNN);
		}
		
		for (j = 0; j < dim; j++)
		{
			for (p = 0; p < N; p++)
			{
				mpc_fma(s[i].poly[p], z[j].poly[p], b->mat[j][i].poly[p], s[i].poly[p], MPC_RNDNN);
			}
		}
	}
	
	for (i = 0; i < dim; i++)
	{
		poly_fft_clear(t + i, N);
		poly_fft_clear(z + i, N);
	}
	
	mpc_clear(tmp1);
	mpc_clear(tmp2);
}
