/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Keygen                         *
 * ****************************** */
 
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "keygen.h"
#include "param.h"
#include "fft.h"
#include "ntt.h"
#include "sample_z.h"
#include "poly.h"
#include "mat.h"
#include "fastrandombytes.h"
#include "red.h"

#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>

#define SAMPLE_B_LEN 1083
#define SAMPLE_B_BYTE 3
#define SAMPLE_B_BOUND 16760833

/* max(||g, -f||, ||qf* / (f * f* + g * g*), qg* / (f * f* + g * g*)) ?> \sigma_0 * \sqrt(2N) */
static int64_t gs_norm(const POLY_64 *f, const POLY_64 *g, const mpfr_t norm_bound)
{
	static POLY_FFT fft_f, fft_g;
	mpc_t denom;
	mpfr_t norm2;
	mpfr_t tmp;
	
	mpc_t fft_f_adj, fft_g_adj;
	
	uint64_t i;
	int64_t ret;
	
	uint64_t norm1 = 0;
	
	for (i = 0; i < N; i++)
	{
		norm1 += f->poly[i] * f->poly[i] + g->poly[i] * g->poly[i];
	}
	
	ret = mpfr_cmp_ui(norm_bound, norm1);
	
	if (ret >= 0)
	{
		poly_fft_init(&fft_f, N);
		poly_fft_init(&fft_g, N);
		
		for (i = 0; i < N; i++)
		{
			mpc_set_si(fft_f.poly[i], f->poly[i], MPC_RNDNN);
			mpc_set_si(fft_g.poly[i], g->poly[i], MPC_RNDNN);
		}
		
		mpc_init2(denom, PREC);
		mpc_init2(fft_f_adj, PREC);
		mpc_init2(fft_g_adj, PREC);
		
		mpfr_inits2(PREC, norm2, tmp, NULL);
				
		fft(&fft_f, N);
		fft(&fft_g, N);
		
		for (i = 0; i < N; i++)
		{
			mpc_conj(fft_f_adj, fft_f.poly[i], MPC_RNDNN);
			mpc_mul(denom, fft_f.poly[i], fft_f_adj, MPC_RNDNN);
			
			mpc_conj(fft_g_adj, fft_g.poly[i], MPC_RNDNN);
			mpc_fma(denom, fft_g.poly[i], fft_g_adj, denom, MPC_RNDNN);
			
			mpc_div(fft_f.poly[i], fft_f_adj, denom, MPC_RNDNN);
			mpc_div(fft_g.poly[i], fft_g_adj, denom, MPC_RNDNN);
		}
		
		ifft(&fft_f, N);
		ifft(&fft_g, N);
		
		mpfr_set_zero(norm2, 0);
		for (i = 0; i < N; i++)
		{
			mpfr_sqr(tmp, mpc_realref(fft_f.poly[i]), MPFR_RNDN);
			mpfr_add(norm2, norm2, tmp, MPFR_RNDN);
			mpfr_sqr(tmp, mpc_realref(fft_g.poly[i]), MPFR_RNDN);
			mpfr_add(norm2, norm2, tmp, MPFR_RNDN);
		}
		
		mpfr_mul_ui(norm2, norm2, Q, MPFR_RNDN);
		mpfr_mul_ui(norm2, norm2, Q, MPFR_RNDN);
		
		ret = mpfr_greater_p(norm2, norm_bound);
		
		poly_fft_clear(&fft_f, N);
		poly_fft_clear(&fft_g, N);
		
		mpc_clear(denom);
		mpc_clear(fft_f_adj);
		mpc_clear(fft_g_adj);
		
		mpfr_clears(norm2, tmp, NULL);
	}
	else
	{
		ret = 1;
	}

	return ret;
}

/* field norm N(a) from NTRUSolve in Falcon */
static void field_norm(POLY_Z *out, const POLY_Z *a, const uint64_t n)
{
	static POLY_Z a_e, a_o;
	static POLY_Z tmp;
	
	uint64_t i;
	
	poly_z_init(&a_e, n);
	poly_z_init(&a_o, n);
	poly_z_init(&tmp, n);
	
	for (i = 0; i < n; i++)
	{
		mpz_set(a_e.poly[i], a->poly[i << 1]);
		mpz_set(a_o.poly[i], a->poly[(i << 1) + 1]);
	}
	
	poly_z_mul(out, &a_e, &a_e, n);
	poly_z_mul(&tmp, &a_o, &a_o, n);
	
	mpz_add(out->poly[0], out->poly[0], tmp.poly[n - 1]);
	for (i = 1; i < n; i++)
	{
		mpz_sub(out->poly[i], out->poly[i], tmp.poly[i - 1]);
	}
	
	poly_z_clear(&a_e, n);
	poly_z_clear(&a_o, n);
	poly_z_clear(&tmp, n);
}

/* g^{x}(x)F'(x^2) from NTRUSolve in Falcon */
static void lift(POLY_Z *out, const POLY_Z *g, const POLY_Z *F_prime, const uint64_t n)
{
	static POLY_Z gx, F_prime_x2;
	
	uint64_t i;
	
	poly_z_init(&gx, n);
	poly_z_init(&F_prime_x2, n);
	
	for (i = 0; i < n; i += 2)
	{
		mpz_set(gx.poly[i], g->poly[i]);
		mpz_neg(gx.poly[i + 1], g->poly[i + 1]);
	}
	
	for (i = 0; i < (n >> 1); i++)
	{
		mpz_set(F_prime_x2.poly[i << 1], F_prime->poly[i]);
	}
	
	poly_z_mul(out, &gx, &F_prime_x2, n);
	
	poly_z_clear(&gx, n);
	poly_z_clear(&F_prime_x2, n);
}

/* length reduction */
static void reduce_k(POLY_Z *F_red, POLY_Z *G_red, const POLY_Z *f, const POLY_Z *g, const POLY_Z *F, const POLY_Z *G, const uint64_t n)
{
	uint64_t i;
	
	static POLY_FFT f_hi, g_hi;
	static POLY_FFT F_hi, G_hi;
	
	static POLY_FFT k_denom;
	static POLY_FFT k_fft;
	
	static POLY_FFT f_hi_adj, g_hi_adj;
	
	mpfr_t k_round;
	static POLY_Z k_poly, fk, gk;
	
	uint64_t check;
	
	poly_fft_init(&f_hi, n);
	poly_fft_init(&g_hi, n);
	
	poly_fft_init(&f_hi_adj, n);
	poly_fft_init(&g_hi_adj, n);
	
	poly_fft_init(&k_denom, n);
	
	poly_fft_init(&k_fft, n);
	
	poly_fft_init(&F_hi, n);
	poly_fft_init(&G_hi, n);
	
	for (i = 0; i < n; i++)
	{
		mpc_set_z(f_hi.poly[i], f->poly[i], MPC_RNDNN);
		mpc_set_z(g_hi.poly[i], g->poly[i], MPC_RNDNN);
	}
	
	fft(&f_hi, n);
	fft(&g_hi, n);
	
	for (i = 0; i < n; i++)
	{
		mpc_conj(f_hi_adj.poly[i], f_hi.poly[i], MPC_RNDNN);
		mpc_conj(g_hi_adj.poly[i], g_hi.poly[i], MPC_RNDNN);

		mpc_mul(k_denom.poly[i], f_hi.poly[i], f_hi_adj.poly[i], MPC_RNDNN);
		mpc_fma(k_denom.poly[i], g_hi.poly[i], g_hi_adj.poly[i], k_denom.poly[i], MPC_RNDNN);
	}
	
	for (i = 0; i < n; i++)
	{
		mpz_set(F_red->poly[i], F->poly[i]);
		mpz_set(G_red->poly[i], G->poly[i]);
	}
	
	mpfr_init2(k_round, PREC);
	poly_z_init(&k_poly, n);
	
	poly_z_init(&fk, n);
	poly_z_init(&gk, n);
	
	while (1)
	{
		for (i = 0; i < n; i++)
		{
			mpc_set_z(F_hi.poly[i], F_red->poly[i], MPC_RNDNN);
			mpc_set_z(G_hi.poly[i], G_red->poly[i], MPC_RNDNN);
		}
		
		fft(&F_hi, n);
		fft(&G_hi, n);
		
		for (i = 0; i < n; i++)
		{
			mpc_mul(k_fft.poly[i], F_hi.poly[i], f_hi_adj.poly[i], MPC_RNDNN);
			mpc_fma(k_fft.poly[i], G_hi.poly[i], g_hi_adj.poly[i], k_fft.poly[i], MPC_RNDNN);
			
			mpc_div(k_fft.poly[i], k_fft.poly[i], k_denom.poly[i], MPC_RNDNN);
		}
		
		ifft(&k_fft, n);
		
		check = 0;
		for (i = 0; i < n; i++)
		{
			mpfr_round(k_round, mpc_realref(k_fft.poly[i]));
			mpfr_get_z(k_poly.poly[i], k_round, MPFR_RNDN);
			
			check |= mpz_sgn(k_poly.poly[i]);
		}
		
		if (!check)
		{
			break;
		}
		
		poly_z_mul(&fk, f, &k_poly, n);
		poly_z_mul(&gk, g, &k_poly, n);
		
		for (i = 0; i < n; i++)
		{
			mpz_sub(F_red->poly[i], F_red->poly[i], fk.poly[i]);
			mpz_sub(G_red->poly[i], G_red->poly[i], gk.poly[i]);
		}
	}
	
	poly_fft_clear(&f_hi, n);
	poly_fft_clear(&g_hi, n);
	
	poly_fft_clear(&f_hi_adj, n);
	poly_fft_clear(&g_hi_adj, n);
	
	poly_fft_clear(&k_denom, n);
	
	poly_fft_clear(&k_fft, n);
	
	poly_fft_clear(&F_hi, n);
	poly_fft_clear(&G_hi, n);

	mpfr_clear(k_round);
	
	poly_z_clear(&k_poly, n);
	poly_z_clear(&fk, n);
	poly_z_clear(&gk, n);
}

/* NTRUSolve from Falcon */
static int64_t tower_solver(POLY_Z *F, POLY_Z *G, const POLY_Z *f, const POLY_Z *g, const uint64_t n)
{
	mpz_t d, u, v;
	
	int64_t ret;
	
	uint64_t n2 = n >> 1;

	static POLY_Z F_prime, G_prime, F_unred, G_unred;
	static uint64_t initialised;
	
	POLY_Z f_prime, g_prime;
	
	if (n == 1)
	{
		mpz_inits(d, u, v, NULL);
		
		mpz_gcdext(d, u, v, f->poly[0], g->poly[0]);
		
		if (mpz_cmp_ui(d, 1))
		{
			ret = 1;
		}
		else
		{
			mpz_mul_si(F->poly[0], v, -Q);
			mpz_mul_ui(G->poly[0], u, Q);
			
			ret = 0;
		}
		
		mpz_clears(d, u, v, NULL);
		
		return ret;
	}
	else
	{
		if (!initialised)
		{
			poly_z_init(&F_prime, N >> 1);
			poly_z_init(&G_prime, N >> 1);

			poly_z_init(&F_unred, N);
			poly_z_init(&G_unred, N);
			
			initialised = 1;
		}
				
		poly_z_init(&f_prime, n2);
		poly_z_init(&g_prime, n2);
		
		field_norm(&f_prime, f, n2);
		field_norm(&g_prime, g, n2);
		
		ret = tower_solver(&F_prime, &G_prime, &f_prime, &g_prime, n2);
		
		if (!ret)
		{
			lift(&F_unred, g, &F_prime, n);
			lift(&G_unred, f, &G_prime, n);
			
			reduce_k(F, G, f, g, &F_unred, &G_unred, n);
		}
		
		poly_z_clear(&f_prime, n2);
		poly_z_clear(&g_prime, n2);
		
		if (n == N)
		{
			poly_z_clear(&F_prime, N >> 1);
			poly_z_clear(&G_prime, N >> 1);

			poly_z_clear(&F_unred, N);
			poly_z_clear(&G_unred, N);
			
			initialised = 0;
		}
		
		return ret;
	}
}

/* Generate (f, g, F, G) such that f * G - g * F = q */
static void ntru_basis(POLY_64 *f, POLY_64 *g, POLY_64 *F, POLY_64 *G)
{
	static POLY_Z f_z, g_z, F_z, G_z;
	
	mpfr_t norm_bound;
	uint64_t i;
	
	poly_z_init(&f_z, N);
	poly_z_init(&g_z, N);
	poly_z_init(&F_z, N);
	poly_z_init(&G_z, N);
	
	mpfr_init2(norm_bound, PREC);
	
	mpfr_set_str(norm_bound, norm_str[0], 10, MPFR_RNDN);
		
	while (1)
	{
		/* f, g <-- (D_{\sigma_0})^N */
		sample_0z(f);
		sample_0z(g);
		
		/* Norm check */
		if (gs_norm(f, g, norm_bound))
		{
			continue;
		}
		
		for (i = 0; i < N; i++)
		{
			mpz_set_si(f_z.poly[i], f->poly[i]);
			mpz_set_si(g_z.poly[i], g->poly[i]);
		}
		
		/* Find F, G such that f * G - g * F = q */
		if (tower_solver(&F_z, &G_z, &f_z, &g_z, N))
		{
			continue;
		}
		
		break;
	}
	
	for (i = 0; i < N; i++)
	{
		F->poly[i] = mpz_get_si(F_z.poly[i]);
		G->poly[i] = mpz_get_si(G_z.poly[i]);
	}
	
	mpfr_clear(norm_bound);

	poly_z_clear(&f_z, N);
	poly_z_clear(&g_z, N);
	poly_z_clear(&F_z, N);
	poly_z_clear(&G_z, N);
}

static inline uint64_t load_24(const unsigned char *x)
{
	return ((uint64_t)(*x)) | (((uint64_t)(*(x + 1))) << 8) | (((uint64_t)(*(x + 2))) << 16);
}

static void sample_b(POLY_64 *b)
{
	static unsigned char r[SAMPLE_B_LEN * SAMPLE_B_BYTE];
	
	uint64_t i, x;
	unsigned char *r_head = r;
	
	fastrandombytes(r, SAMPLE_B_LEN * SAMPLE_B_BYTE);
	
	for (i = 0; i < N; i++)
	{
		do
		{
			x = load_24(r_head);
			r_head += SAMPLE_B_BYTE;
		} while (x >= SAMPLE_B_BOUND);
		
		b->poly[i] = x;
	}
}

void keygen(MAT_64 *basis, POLY_64 *h, POLY_64 *b, const unsigned char *seed)
{
	static POLY_64 f_ntt, g_ntt;
	
	uint64_t i;
	
	fastrandombytes_setseed(seed);
	
	ntru_basis(&(basis->mat[0][1]), &(basis->mat[0][0]), &(basis->mat[1][1]), &(basis->mat[1][0]));
	
	memcpy(&f_ntt, &(basis->mat[0][1]), sizeof(POLY_64));
	memcpy(&g_ntt, &(basis->mat[0][0]), sizeof(POLY_64));

	ntt(&f_ntt);
	ntt(&g_ntt);
	
	/* h = g * f^{-1} mod q */
	for (i = 0; i < N; i++)
	{
		h->poly[i] = montgomery(montgomery(g_ntt.poly[i], inverse(f_ntt.poly[i])), MONTGOMERY_INV_FACTOR);
	}
	
	/* g | -f
	 * ------
	 * G | -F */
	for (i = 0; i < N; i++)
	{
		basis->mat[0][1].poly[i] = -basis->mat[0][1].poly[i];
		basis->mat[1][1].poly[i] = -basis->mat[1][1].poly[i];
	}
	
	/* b <-- U(R_q) */
	sample_b(b);
}
