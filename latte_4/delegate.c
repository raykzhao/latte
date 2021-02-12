/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Delegation                     *
 * ****************************** */

#include <stdint.h>
#include "delegate.h"
#include "param.h"
#include "poly.h"
#include "mat.h"
#include "sample_z.h"
#include "ntt.h"
#include "fft.h"
#include "sample_fft.h"
#include "keygen.h"
#include "fastrandombytes.h"
#include "red.h"

#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>

void delegate(MAT_64 *s, const MAT_64 *basis, const POLY_64 *a, const uint64_t l, const unsigned char *seed)
{
	static MAT_FFT fft_basis;
	static MAT_FFT basis_g;
	static MAT_FFT tree_root;
	static POLY_FFT tree_dim2[L * (N - 1)];
	
	static POLY_64 c_ntt;
	static POLY_FFT c;
	static MAT_FFT s_fft;
	static POLY_FFT s_ifft;
	
	static POLY_FFT f_fft, g_fft;
	static POLY_Z f, g, F, G;
	
	static MAT_FFT s_fft_head;
	
	static MAT_FFT s_g;
	static POLY_FFT s_g_det;
	
	static POLY_FFT F_fft, G_fft;
	static POLY_FFT d[L];
	static POLY_FFT k_red_fft;
	static POLY_Z k_red;
	static POLY_Z s_z, ks, z;
	
	uint64_t i, j, k, p;
	
	mpfr_t sigma;
	mpfr_t center;
	mpfr_t norm_bound;
	
	mpfr_t tmp;
	mpc_t tmp2;
	
	uint64_t norm;
	
	uint64_t f_det_check;
	
	fastrandombytes_setseed(seed);
	
	mat_fft_init(&fft_basis, l + 1, N);
	mat_fft_init(&basis_g, l + 1, N);
	
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
	
	mat_fft_init(&s_fft, l + 2, N);
	
	poly_fft_init(&s_ifft, N);
	
	poly_fft_init(&f_fft, N);
	poly_fft_init(&g_fft, N);

	poly_z_init(&f, N);
	poly_z_init(&g, N);
	poly_z_init(&F, N);
	poly_z_init(&G, N);
	
	mat_fft_init(&s_fft_head, l + 2, N);
	
	mat_fft_init(&s_g, l + 1, N);
	
	poly_fft_init(&s_g_det, N);
	
	poly_fft_init(&F_fft, N);
	poly_fft_init(&G_fft, N);
	
	for (i = 0; i < l + 1; i++)
	{
		poly_fft_init(d + i, N);
	}
	
	poly_fft_init(&k_red_fft, N);
	poly_z_init(&k_red, N);
	
	poly_z_init(&s_z, N);
	poly_z_init(&ks, N);
	poly_z_init(&z, N);
	
	mpfr_inits2(PREC, sigma, center, norm_bound, tmp, NULL);
	mpc_init2(tmp2, PREC);
	
	mpfr_set_str(sigma, sigma_str[l], 10, MPFR_RNDN);
	mpfr_set_str(norm_bound, norm_str[l], 10, MPFR_RNDN);
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
	
	gram(&basis_g, &fft_basis, l + 1, N);
	
	fft_ldl(&tree_root, tree_dim2, &basis_g, l + 1, sigma);
	
	do
	{
		for (i = 0; i < l + 1; i++)
		{
			do
			{
				norm = 0;
				
				for (p = 0; p < N; p++)
				{
					/* s_{i, l + 1} <-- (D_{\sigma_l})^N */
					s->mat[i][l + 1].poly[p] = sample_z(center, sigma);
					
					c_ntt.poly[p] = -(s->mat[i][l + 1].poly[p]);
					
					norm += s->mat[i][l + 1].poly[p] * s->mat[i][l + 1].poly[p];
				}
				
				ntt(&c_ntt);
				
				/* c = -s_{i, l + 1} * A_l */
				for (p = 0; p < N; p++)
				{
					c_ntt.poly[p] = montgomery(c_ntt.poly[p], a->poly[p]);
				}
				
				intt(&c_ntt);
				
				for (p = 0; p < N; p++)
				{
					mpc_set_ui(c.poly[p], c_ntt.poly[p], MPC_RNDNN);
				}
				
				fft(&c, N);
				
				/* sample preimage */
				sample_preimage(s_fft.mat[i], &fft_basis, &tree_root, tree_dim2, &c, l + 1);
				
				for (j = 0; j < l + 1; j++)
				{
					for (p = 0; p < N; p++)
					{
						mpc_set(s_ifft.poly[p], s_fft.mat[i][j].poly[p], MPC_RNDNN);
					}
					
					ifft(&s_ifft, N);
					
					for (p = 0; p < N; p++)
					{
						mpfr_round(tmp, mpc_realref(s_ifft.poly[p]));
						
						s->mat[i][j].poly[p] = mpfr_get_si(tmp, MPFR_RNDN);
						
						norm += s->mat[i][j].poly[p] * s->mat[i][j].poly[p];
					}
				}
				/* ||(s_{i, 0},...,s_{i, l + 1}|| ?> \sqrt{(l + 2)N} * \sigma_l */
			} while (mpfr_cmp_ui(norm_bound, norm) < 0);
			
			for (p = 0; p < N; p++)
			{
				mpc_set_si(s_fft.mat[i][l + 1].poly[p], s->mat[i][l + 1].poly[p], MPC_RNDNN);
			}
			
			fft(&(s_fft.mat[i][l + 1]), N);
		}
		
		/* ModFalcon method to solve NTRU euqation for dim > 2 
		 * Here we only consider l = 1 */
		if (l == 1)
		{
			for (p = 0; p < N; p++)
			{
				mpc_mul(f_fft.poly[p], s_fft.mat[0][1].poly[p], s_fft.mat[1][2].poly[p], MPC_RNDNN);
				mpc_mul(tmp2, s_fft.mat[0][2].poly[p], s_fft.mat[1][1].poly[p], MPC_RNDNN);
				mpc_sub(f_fft.poly[p], f_fft.poly[p], tmp2, MPC_RNDNN);
				
				mpc_mul(g_fft.poly[p], s_fft.mat[0][0].poly[p], s_fft.mat[1][2].poly[p], MPC_RNDNN);
				mpc_mul(tmp2, s_fft.mat[0][2].poly[p], s_fft.mat[1][0].poly[p], MPC_RNDNN);
				mpc_sub(g_fft.poly[p], g_fft.poly[p], tmp2, MPC_RNDNN);
			}
		}
		
		ifft(&f_fft, N);
		ifft(&g_fft, N);
		
		f_det_check = 0;
		for (p = 0; p < N; p++)
		{
			mpfr_round(tmp, mpc_realref(f_fft.poly[p]));
			mpfr_get_z(f.poly[p], tmp, MPFR_RNDN);
			
			f_det_check |= mpz_sgn(f.poly[p]);
			
			mpfr_round(tmp, mpc_realref(g_fft.poly[p]));
			mpfr_get_z(g.poly[p], tmp, MPFR_RNDN);
		}
		
		if (!f_det_check)
		{
			continue;
		}
	} while (tower_solver(&F, &G, &f, &g, N));
	
	for (i = 0; i < l + 1; i++)
	{
		for (j = 0; j < l + 2; j++)
		{
			for (p = 0; p < N; p++)
			{
				mpc_conj(s_fft_head.mat[i][j].poly[p], s_fft.mat[i][j].poly[p], MPC_RNDNN);
			}
		}
	}
	
	/* Since the inf-norm of the output in ModFalcon method is about the magnitude of q^{l+1}, we need the further length reduction by using Cramer's rule */
	for (i = 0; i < l + 1; i++)
	{
		for (j = 0; j < l + 1; j++)
		{
			for (p = 0; p < N; p++)
			{
				mpc_set_ui(s_g.mat[i][j].poly[p], 0, MPC_RNDNN);
			}
			
			for (k = 0; k < l + 2; k++)
			{
				for (p = 0; p < N; p++)
				{
					mpc_fma(s_g.mat[i][j].poly[p], s_fft_head.mat[i][k].poly[p], s_fft.mat[j][k].poly[p], s_g.mat[i][j].poly[p], MPC_RNDNN);
				}
			}
		}
	}
	
	/* Here we only consider l = 1 */
	if (l == 1)
	{
		for (p = 0; p < N; p++)
		{
			mpc_mul(s_g_det.poly[p], s_g.mat[0][0].poly[p], s_g.mat[1][1].poly[p], MPC_RNDNN);
			mpc_mul(tmp2, s_g.mat[0][1].poly[p], s_g.mat[1][0].poly[p], MPC_RNDNN);
			mpc_sub(s_g_det.poly[p], s_g_det.poly[p], tmp2, MPC_RNDNN);
		}
	}
	
	for (p = 0; p < N; p++)
	{
		mpc_set_z(F_fft.poly[p], F.poly[p], MPC_RNDNN);
		mpc_set_z(G_fft.poly[p], G.poly[p], MPC_RNDNN);
	}
	
	fft(&F_fft, N);
	fft(&G_fft, N);
	
	for (i = 0; i < l + 1; i++)
	{
		for (p = 0; p < N; p++)
		{
			mpc_mul(d[i].poly[p], s_fft_head.mat[i][0].poly[p], G_fft.poly[p], MPC_RNDNN);
			mpc_fma(d[i].poly[p], s_fft_head.mat[i][1].poly[p], F_fft.poly[p], d[i].poly[p], MPC_RNDNN);
		}
	}
	
	/* Here we only consider l = 1 */
	if (l == 1)
	{
		for (p = 0; p < N; p++)
		{
			mpc_mul(k_red_fft.poly[p], d[0].poly[p], s_g.mat[1][1].poly[p], MPC_RNDNN);
			mpc_mul(tmp2, s_g.mat[0][1].poly[p], d[1].poly[p], MPC_RNDNN);
			mpc_sub(k_red_fft.poly[p], k_red_fft.poly[p], tmp2, MPC_RNDNN);
			mpc_div(k_red_fft.poly[p], k_red_fft.poly[p], s_g_det.poly[p], MPC_RNDNN);
		}
		
		ifft(&k_red_fft, N);
		
		for (p = 0; p < N; p++)
		{
			mpfr_round(tmp, mpc_realref(k_red_fft.poly[p]));
			mpfr_get_z(k_red.poly[p], tmp, MPFR_RNDN);
		}
				
		for (p = 0; p < N; p++)
		{
			mpz_set_si(s_z.poly[p], s->mat[0][0].poly[p]);
		}
		poly_z_mul(&ks, &k_red, &s_z, N);
		for (p = 0; p < N; p++)
		{
			mpz_sub(G.poly[p], G.poly[p], ks.poly[p]);
		}

		for (p = 0; p < N; p++)
		{
			mpz_set_si(s_z.poly[p], s->mat[0][1].poly[p]);
		}
		poly_z_mul(&ks, &k_red, &s_z, N);
		for (p = 0; p < N; p++)
		{
			mpz_sub(F.poly[p], F.poly[p], ks.poly[p]);
		}

		for (p = 0; p < N; p++)
		{
			mpz_set_si(s_z.poly[p], s->mat[0][2].poly[p]);
		}
		poly_z_mul(&ks, &k_red, &s_z, N);
		for (p = 0; p < N; p++)
		{
			mpz_neg(z.poly[p], ks.poly[p]);
		}
		
		for (p = 0; p < N; p++)
		{
			mpc_mul(k_red_fft.poly[p], s_g.mat[0][0].poly[p], d[1].poly[p], MPC_RNDNN);
			mpc_mul(tmp2, d[0].poly[p], s_g.mat[1][0].poly[p], MPC_RNDNN);
			mpc_sub(k_red_fft.poly[p], k_red_fft.poly[p], tmp2, MPC_RNDNN);
			mpc_div(k_red_fft.poly[p], k_red_fft.poly[p], s_g_det.poly[p], MPC_RNDNN);
		}
		
		ifft(&k_red_fft, N);
		
		for (p = 0; p < N; p++)
		{
			mpfr_round(tmp, mpc_realref(k_red_fft.poly[p]));
			mpfr_get_z(k_red.poly[p], tmp, MPFR_RNDN);
		}
		
		for (p = 0; p < N; p++)
		{
			mpz_set_si(s_z.poly[p], s->mat[1][0].poly[p]);
		}
		poly_z_mul(&ks, &k_red, &s_z, N);
		for (p = 0; p < N; p++)
		{
			mpz_sub(G.poly[p], G.poly[p], ks.poly[p]);
		}

		for (p = 0; p < N; p++)
		{
			mpz_set_si(s_z.poly[p], s->mat[1][1].poly[p]);
		}
		poly_z_mul(&ks, &k_red, &s_z, N);
		for (p = 0; p < N; p++)
		{
			mpz_sub(F.poly[p], F.poly[p], ks.poly[p]);
		}

		for (p = 0; p < N; p++)
		{
			mpz_set_si(s_z.poly[p], s->mat[1][2].poly[p]);
		}
		poly_z_mul(&ks, &k_red, &s_z, N);
		for (p = 0; p < N; p++)
		{
			mpz_sub(z.poly[p], z.poly[p], ks.poly[p]);
		}
		
		for (p = 0; p < N; p++)
		{
			s->mat[2][0].poly[p] = mpz_get_si(G.poly[p]);
			s->mat[2][1].poly[p] = mpz_get_si(F.poly[p]);
			s->mat[2][2].poly[p] = mpz_get_si(z.poly[p]);			
		}
	}
	
	mat_fft_clear(&fft_basis, l + 1, N);
	mat_fft_clear(&basis_g, l + 1, N);
	
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
	
	mat_fft_clear(&s_fft, l + 2, N);
	
	poly_fft_clear(&s_ifft, N);
	
	poly_fft_clear(&f_fft, N);
	poly_fft_clear(&g_fft, N);
	
	poly_z_clear(&f, N);
	poly_z_clear(&g, N);
	poly_z_clear(&F, N);
	poly_z_clear(&G, N);
	
	mat_fft_clear(&s_fft_head, l + 2, N);
	
	mat_fft_clear(&s_g, l + 1, N);
	
	poly_fft_clear(&s_g_det, N);
	
	poly_fft_clear(&F_fft, N);
	poly_fft_clear(&G_fft, N);
	
	for (i = 0; i < l + 1; i++)
	{
		poly_fft_clear(d + i, N);
	}
	
	poly_fft_clear(&k_red_fft, N);
	poly_z_clear(&k_red, N);

	poly_z_clear(&s_z, N);
	poly_z_clear(&ks, N);
	poly_z_clear(&z, N);
		
	mpfr_clears(sigma, center, norm_bound, tmp, NULL);
	mpc_clear(tmp2);
}
