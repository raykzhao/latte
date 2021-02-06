/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Integer samplers               *
 * ****************************** */
 
#include <stdint.h>
#include "sample_z.h"
#include "param.h"
#include "fastrandombytes.h"
#include "poly.h"

#include <mpfr.h>

static const char pi2_str[PREC] = "6.283185307179586476925286766559005768394338798750211641949889184615632812572";
static const char sqrt_pi2_str[PREC] = "2.506628274631000502415765284811045253006986740609938316629923576342293654608";

static mpfr_t pi2;
static mpfr_t sqrt_pi2;

#define BOX_MULLER_BYTES (PREC * 2 / 8)
#define COMP_ENTRY_SIZE (PREC / 8)
#define DISCRETE_BYTES (1 + 2 * COMP_ENTRY_SIZE)

static uint64_t sample_z_initialised;

static void sample_z_init()
{
	if (!sample_z_initialised)
	{
		mpfr_inits2(PREC, pi2, sqrt_pi2, NULL);
		
		mpfr_set_str(pi2, pi2_str, 10, MPFR_RNDN);
		mpfr_set_str(sqrt_pi2, sqrt_pi2_str, 10, MPFR_RNDN);
		
		sample_z_initialised = 1;
	}
}

/* COSAC sampler */
int64_t sample_z(const mpfr_t center, const mpfr_t sigma)
{
	unsigned char r[DISCRETE_BYTES]; 	
	
	mpfr_t c, cr, rc;
	mpfr_t yr, rej;
	mpfr_t sigma2;
	mpfr_t discrete_normalisation;
	mpfr_t comp;
	mpfr_t yrc;
	
	uint64_t b, i;
	int64_t cmp1;
	uint64_t head = 2;

	uint64_t r_bm[BOX_MULLER_BYTES / 8];

	mpfr_t r1, r2;
	mpfr_t norm[2];
	
	int64_t ret;
	
	sample_z_init();
	
	mpfr_inits2(PREC, c, cr, rc, yr, rej, sigma2, discrete_normalisation, comp, r1, r2, norm[0], norm[1], yrc, NULL);
	
	mpfr_round(cr, center);
	mpfr_sub(c, cr, center, MPFR_RNDN);
	
	mpfr_sqr(sigma2, sigma, MPFR_RNDN);
	mpfr_mul_2ui(sigma2, sigma2, 1, MPFR_RNDN);
	mpfr_neg(sigma2, sigma2, MPFR_RNDN);
	
	mpfr_mul(discrete_normalisation, sigma, sqrt_pi2, MPFR_RNDN);
	
	mpfr_sqr(rc, c, MPFR_RNDN);
	mpfr_div(rc, rc, sigma2, MPFR_RNDN);
	mpfr_exp(rc, rc, MPFR_RNDN);
	mpfr_div(rc, rc, discrete_normalisation, MPFR_RNDN);
	
	fastrandombytes(r, COMP_ENTRY_SIZE);
	
	mpfr_set_ui(comp, ((uint64_t *)r)[3], MPFR_RNDN);
	mpfr_mul_2ui(comp, comp, 64, MPFR_RNDN);
	mpfr_add_ui(comp, comp, ((uint64_t *)r)[2], MPFR_RNDN);
	mpfr_mul_2ui(comp, comp, 64, MPFR_RNDN);
	mpfr_add_ui(comp, comp, ((uint64_t *)r)[1], MPFR_RNDN);
	mpfr_mul_2ui(comp, comp, 64, MPFR_RNDN);
	mpfr_add_ui(comp, comp, ((uint64_t *)r)[0], MPFR_RNDN);
	mpfr_div_2ui(comp, comp, PREC, MPFR_RNDN);
	
	if (mpfr_less_p(comp, rc))
	{
		ret = mpfr_get_si(cr, MPFR_RNDN);

		mpfr_clears(c, cr, rc, yr, rej, sigma2, discrete_normalisation, comp, r1, r2, norm[0], norm[1], yrc, NULL);
		return ret;
	}
	
	while (1)
	{
		fastrandombytes(r, DISCRETE_BYTES);

		for (i = 0; i < 2; i++, head++)
		{
			if (head >= 2)
			{
				head = 0;
				
				fastrandombytes((unsigned char *)r_bm, BOX_MULLER_BYTES);
				
				mpfr_set_ui(r1, r_bm[3], MPFR_RNDN);
				mpfr_mul_2ui(r1, r1, 64, MPFR_RNDN);
				mpfr_add_ui(r1, r1, r_bm[2], MPFR_RNDN);
				mpfr_mul_2ui(r1, r1, 64, MPFR_RNDN);
				mpfr_add_ui(r1, r1, r_bm[1], MPFR_RNDN);
				mpfr_mul_2ui(r1, r1, 64, MPFR_RNDN);
				mpfr_add_ui(r1, r1, r_bm[0], MPFR_RNDN);
				mpfr_div_2ui(r1, r1, PREC, MPFR_RNDN);
				
				mpfr_set_ui(r2, r_bm[7], MPFR_RNDN);
				mpfr_mul_2ui(r2, r2, 64, MPFR_RNDN);
				mpfr_add_ui(r2, r2, r_bm[6], MPFR_RNDN);
				mpfr_mul_2ui(r2, r2, 64, MPFR_RNDN);
				mpfr_add_ui(r2, r2, r_bm[5], MPFR_RNDN);
				mpfr_mul_2ui(r2, r2, 64, MPFR_RNDN);
				mpfr_add_ui(r2, r2, r_bm[4], MPFR_RNDN);
				mpfr_div_2ui(r2, r2, PREC, MPFR_RNDN);
				
				mpfr_log(r1, r1, MPFR_RNDN);
				mpfr_mul_si(r1, r1, -2, MPFR_RNDN);
				mpfr_sqrt(r1, r1, MPFR_RNDN);
				mpfr_mul(r1, r1, sigma, MPFR_RNDN);
				
				mpfr_mul(r2, r2, pi2, MPFR_RNDN);
				
				mpfr_sin_cos(norm[0], norm[1], r2, MPFR_RNDN);
				
				mpfr_mul(norm[0], r1, norm[0], MPFR_RNDN);
				mpfr_mul(norm[1], r1, norm[1], MPFR_RNDN);
			}

			b = (r[DISCRETE_BYTES - 1] >> i) & 0x01;
			
			mpfr_round(yr, norm[head]);
			if (b)
			{
				mpfr_add_ui(yr, yr, 1, MPFR_RNDN);
				cmp1 = mpfr_cmp_d(norm[head], -0.5) >= 0;
			}
			else
			{
				mpfr_sub_ui(yr, yr, 1, MPFR_RNDN);
				cmp1 = mpfr_cmp_d(norm[head], 0.5) <= 0;
			}
			
			if (cmp1)
			{
				mpfr_add(yrc, yr, c, MPFR_RNDN);
				mpfr_add(rej, yrc, norm[head], MPFR_RNDN);
				mpfr_sub(yrc, yrc, norm[head], MPFR_RNDN);
				mpfr_mul(rej, rej, yrc, MPFR_RNDN);
				mpfr_div(rej, rej, sigma2, MPFR_RNDN);
				mpfr_exp(rej, rej, MPFR_RNDN);
				
				mpfr_set_ui(comp, ((uint64_t *)r)[i * (COMP_ENTRY_SIZE / 8) + 3], MPFR_RNDN);
				mpfr_mul_2ui(comp, comp, 64, MPFR_RNDN);
				mpfr_add_ui(comp, comp, ((uint64_t *)r)[i * (COMP_ENTRY_SIZE / 8) + 2], MPFR_RNDN);
				mpfr_mul_2ui(comp, comp, 64, MPFR_RNDN);
				mpfr_add_ui(comp, comp, ((uint64_t *)r)[i * (COMP_ENTRY_SIZE / 8) + 1], MPFR_RNDN);
				mpfr_mul_2ui(comp, comp, 64, MPFR_RNDN);
				mpfr_add_ui(comp, comp, ((uint64_t *)r)[i * (COMP_ENTRY_SIZE / 8)], MPFR_RNDN);
				mpfr_div_2ui(comp, comp, PREC, MPFR_RNDN);
				
				if (mpfr_less_p(comp, rej))
				{
					mpfr_add(yr, yr, cr, MPFR_RNDN);
					
					ret = mpfr_get_si(yr, MPFR_RNDN);
					
					mpfr_clears(c, cr, rc, yr, rej, sigma2, discrete_normalisation, comp, r1, r2, norm[0], norm[1], yrc, NULL);
					return ret;
				}
			}
		}
	}	
}

/* sample ephemeral key with st.d \sigma_e = 2 from binomial distribution */
void sample_e(POLY_64 *out)
{
	unsigned char r[N << 1];
	
	uint64_t i, j;
	
	fastrandombytes(r, N << 1);
	
	for (i = 0; i < N; i++)
	{
		out->poly[i] = 0;
		
		for (j = 0; j < 8; j++)
		{
			out->poly[i] += ((r[i << 1] >> j) & 0x1) - ((r[(i << 1) + 1] >> j) & 0x1);
		}
	}
}
