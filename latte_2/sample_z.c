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
#include <x86intrin.h>

/* Constants used by COSAC */
static const char pi2_str[PREC] = "6.283185307179586476925286766559005768394338798750211641949889184615632812572";
static const char sqrt_pi2_str[PREC] = "2.506628274631000502415765284811045253006986740609938316629923576342293654608";

static mpfr_t pi2;
static mpfr_t sqrt_pi2;

#define BOX_MULLER_BYTES (PREC * 2 / 8)
#define COMP_ENTRY_SIZE (PREC / 8)
#define DISCRETE_BYTES (2 * COMP_ENTRY_SIZE)

static uint64_t sample_z_initialised;

/* Constants used by FACCT */
#define CDT_ENTRY_SIZE 16
#define CDT_LOW_MASK 0x7fffffffffffffff
#define CDT_LENGTH 9 /* [0..tau*sigma]=[0..9] */

#define BERNOULLI_ENTRY_SIZE 9 /* 72bit randomness */

/* the closest integer k such that k*sigma_0=sigma */
#define BINARY_SAMPLER_K 125

/* -1/k^2 */
#define BINARY_SAMPLER_K_2_INV (-1.0/(BINARY_SAMPLER_K * BINARY_SAMPLER_K))

#define EXP_MANTISSA_PRECISION 52
#define EXP_MANTISSA_MASK ((1LL << EXP_MANTISSA_PRECISION) - 1)
#define R_MANTISSA_PRECISION (EXP_MANTISSA_PRECISION + 1)
#define R_MANTISSA_MASK ((1LL << R_MANTISSA_PRECISION) - 1)
#define R_EXPONENT_L (8 * BERNOULLI_ENTRY_SIZE - R_MANTISSA_PRECISION)

#define DOUBLE_ONE (1023LL << 52)

#define UNIFORM_SIZE 1
#define UNIFORM_REJ 62
#define BARRETT_BITSHIFT (UNIFORM_SIZE * 8)

#define BARRETT_FACTOR ((1LL << BARRETT_BITSHIFT) / BINARY_SAMPLER_K)
#define UNIFORM_Q (BINARY_SAMPLER_K * BARRETT_FACTOR)

#define BASE_TABLE_SIZE (4 * CDT_ENTRY_SIZE)
#define BERNOULLI_TABLE_SIZE (4 * BERNOULLI_ENTRY_SIZE)

/* CDT table */
static const __m256i V_CDT[][2] = {{{2200310400551559144, 2200310400551559144, 2200310400551559144, 2200310400551559144}, {3327841033070651387, 3327841033070651387, 3327841033070651387, 3327841033070651387}},
{{7912151619254726620, 7912151619254726620, 7912151619254726620, 7912151619254726620}, {380075531178589176, 380075531178589176, 380075531178589176, 380075531178589176}},
{{5167367257772081627, 5167367257772081627, 5167367257772081627, 5167367257772081627}, {11604843442081400, 11604843442081400, 11604843442081400, 11604843442081400}},
{{5081592746475748971, 5081592746475748971, 5081592746475748971, 5081592746475748971}, {90134450315532, 90134450315532, 90134450315532, 90134450315532}},
{{6522074513864805092, 6522074513864805092, 6522074513864805092, 6522074513864805092}, {175786317361, 175786317361, 175786317361, 175786317361}},
{{2579734681240182346, 2579734681240182346, 2579734681240182346, 2579734681240182346}, {85801740, 85801740, 85801740, 85801740}},
{{8175784047440310133, 8175784047440310133, 8175784047440310133, 8175784047440310133}, {10472, 10472, 10472, 10472}},
{{2947787991558061753, 2947787991558061753, 2947787991558061753, 2947787991558061753}, {0, 0, 0, 0}},
{{22489665999543, 22489665999543, 22489665999543, 22489665999543}, {0, 0, 0, 0}}};

static const __m256i V_CDT_LOW_MASK = {CDT_LOW_MASK, CDT_LOW_MASK, CDT_LOW_MASK, CDT_LOW_MASK};

static const __m256i V_K_K_K_K = {BINARY_SAMPLER_K, BINARY_SAMPLER_K, BINARY_SAMPLER_K, BINARY_SAMPLER_K};

/* coefficients of the exp evaluation polynomial */
static const __m256i EXP_COFF[] = {{0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4},
								   {0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1},
								   {0x3ef01b254493363f, 0x3ef01b254493363f, 0x3ef01b254493363f, 0x3ef01b254493363f},
								   {0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc},
								   {0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b},
								   {0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a},
								   {0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e},
								   {0x3fcebfbdff556072, 0x3fcebfbdff556072, 0x3fcebfbdff556072, 0x3fcebfbdff556072},
								   {0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6},
								   {0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000}};
								   
static const __m256d V_INT64_DOUBLE = {0x0010000000000000, 0x0010000000000000, 0x0010000000000000, 0x0010000000000000};
static const __m256d V_DOUBLE_INT64 = {0x0018000000000000, 0x0018000000000000, 0x0018000000000000, 0x0018000000000000};

static const __m256i V_EXP_MANTISSA_MASK = {EXP_MANTISSA_MASK, EXP_MANTISSA_MASK, EXP_MANTISSA_MASK, EXP_MANTISSA_MASK};
static const __m256i V_RES_MANTISSA = {1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION};
static const __m256i V_RES_EXPONENT = {R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1};
static const __m256i V_R_MANTISSA_MASK = {R_MANTISSA_MASK, R_MANTISSA_MASK, R_MANTISSA_MASK, R_MANTISSA_MASK};
static const __m256i V_1 = {1, 1, 1, 1};
static const __m256i V_DOUBLE_ONE = {DOUBLE_ONE, DOUBLE_ONE, DOUBLE_ONE, DOUBLE_ONE};

static const __m256d V_K_2_INV = {BINARY_SAMPLER_K_2_INV, BINARY_SAMPLER_K_2_INV, BINARY_SAMPLER_K_2_INV, BINARY_SAMPLER_K_2_INV};

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

/* New COSAC sampler. 
 * This is the sampling algorithm from:
 * Shuo Sun, Yongbin Zhou, Yunfeng Ji, Rui Zhang, & Yang Tao. (2021). Generic, Efficient and Isochronous Gaussian Sampling over the Integers. 
 * https://eprint.iacr.org/2021/199 */
int64_t sample_z(const mpfr_t center, const mpfr_t sigma)
{
	unsigned char r[DISCRETE_BYTES]; 	
	
	mpfr_t c, cr, rc;
	mpfr_t yr, rej;
	mpfr_t sigma2;
	mpfr_t discrete_normalisation;
	mpfr_t comp;
	mpfr_t yrc;
	
	uint64_t i;
	int64_t cmp1;
	uint64_t head = 2;

	uint64_t r_bm[BOX_MULLER_BYTES / 8];

	mpfr_t r1, r2;
	mpfr_t norm[2];
	
	int64_t ret;
	
	sample_z_init();
	
	mpfr_inits2(PREC, c, cr, rc, yr, rej, sigma2, discrete_normalisation, comp, r1, r2, norm[0], norm[1], yrc, NULL);
	
	mpfr_round(cr, center);
	mpfr_sub(c, center, cr, MPFR_RNDN);
	
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
			
			mpfr_add(yr, norm[head], c, MPFR_RNDN);
			
			cmp1 = mpfr_sgn(yr) >= 0;
			
			mpfr_floor(yr, yr);
			mpfr_add_ui(yr, yr, cmp1, MPFR_RNDN);
			
			mpfr_sub(yrc, yr, c, MPFR_RNDN);
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

/* constant time CDT sampler for \sigma = sqrt(1 / ln(2 * ln(2))) */
static inline __m256i cdt_sampler(unsigned char *r)
{
	__m256i x = _mm256_setzero_si256();
	__m256i r1, r2;
	__m256i r1_lt_cdt0, r2_lt_cdt1;
	__m256i r2_eq_cdt1;
	__m256i b;
	
	uint32_t i;
	
	r1 = _mm256_loadu_si256((__m256i *)r);
	r2 = _mm256_loadu_si256((__m256i *)(r + 32));
	
	r1 = _mm256_and_si256(r1, V_CDT_LOW_MASK);
	r2 = _mm256_and_si256(r2, V_CDT_LOW_MASK);

	for (i = 0; i < CDT_LENGTH; i++)
	{
		r1_lt_cdt0 = _mm256_sub_epi64(r1, V_CDT[i][0]);

		r2_lt_cdt1 = _mm256_sub_epi64(r2, V_CDT[i][1]);
		r2_eq_cdt1 = _mm256_cmpeq_epi64(r2, V_CDT[i][1]);

		b = _mm256_and_si256(r1_lt_cdt0, r2_eq_cdt1);
		b = _mm256_or_si256(b, r2_lt_cdt1);
		b = _mm256_srli_epi64(b, 63);

		x = _mm256_add_epi64(x, b);
	}

	return x;
}

/* constant time Bernoulli sampler
 * we directly compute exp(-x/(2*sigma_0^2)), 
 * since sigma_0=sqrt(1/2ln2), exp(-x/(2*sigma_0^2))=2^(-x/k^2), 
 * we use a polynomial to directly evaluate 2^(-x/k^2) */
static inline void bernoulli_sampler(uint64_t *b, __m256i x, unsigned char *r)
{	
	__m256d vx, vx_1, vx_2, vsum;
	__m256i vt, k, vres, vres_mantissa, vres_exponent, vr_mantissa, vr_exponent, vr_exponent2, vres_eq_1, vr_lt_vres_mantissa, vr_lt_vres_exponent;

	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */
	x = _mm256_or_si256(x, _mm256_castpd_si256(V_INT64_DOUBLE));
	vx = _mm256_sub_pd(_mm256_castsi256_pd(x), V_INT64_DOUBLE);
	vx = _mm256_mul_pd(vx, V_K_2_INV);
	
	vx_1 = _mm256_floor_pd(vx);
	vx_2 = _mm256_add_pd(vx_1, V_DOUBLE_INT64);
	vt = _mm256_sub_epi64(_mm256_castpd_si256(vx_2), _mm256_castpd_si256(V_DOUBLE_INT64));	
	vt = _mm256_slli_epi64(vt, 52);
	
	/* evaluate 2^a */
	vx_2 = _mm256_sub_pd(vx, vx_1);
	vsum = _mm256_fmadd_pd(_mm256_castsi256_pd(EXP_COFF[0]), vx_2, _mm256_castsi256_pd(EXP_COFF[1]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[2]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[3]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[4]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[5]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[6]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[7]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[8]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[9]));
	
	/* combine to compute 2^x */
	vres = _mm256_add_epi64(vt, _mm256_castpd_si256(vsum));

	/* compute the Bernoulli value */
	vres_mantissa = _mm256_and_si256(vres, V_EXP_MANTISSA_MASK);
	vres_mantissa = _mm256_or_si256(vres_mantissa, V_RES_MANTISSA);
	
	vres_exponent = _mm256_srli_epi64(vres, EXP_MANTISSA_PRECISION);
	vres_exponent = _mm256_add_epi64(vres_exponent, V_RES_EXPONENT);
	vres_exponent = _mm256_sllv_epi64(V_1, vres_exponent);
	
	vr_mantissa = _mm256_loadu_si256((__m256i *)r);
	vr_exponent = _mm256_srli_epi64(vr_mantissa, R_MANTISSA_PRECISION);
	vr_mantissa = _mm256_and_si256(vr_mantissa, V_R_MANTISSA_MASK);
	vr_exponent2 = _mm256_set_epi64x(r[35], r[34], r[33], r[32]);
	vr_exponent2 = _mm256_slli_epi64(vr_exponent2, 64 - R_MANTISSA_PRECISION);
	vr_exponent = _mm256_or_si256(vr_exponent, vr_exponent2);

	/* (res == 1.0) || ((r_mantissa < res_mantissa) && (r_exponent < (1 << res_exponent))) */
	vres_eq_1 = _mm256_cmpeq_epi64(vres, V_DOUBLE_ONE);
	vr_lt_vres_mantissa = _mm256_sub_epi64(vr_mantissa, vres_mantissa);	
	vr_lt_vres_exponent = _mm256_sub_epi64(vr_exponent, vres_exponent);
	
	k = _mm256_and_si256(vr_lt_vres_mantissa, vr_lt_vres_exponent);
	k = _mm256_or_si256(k, vres_eq_1);

	_mm256_store_si256((__m256i *)(b), k);
}

/* make sure that Pr(rerun the PRG)<=2^(-256) */
static inline void uniform_sampler(unsigned char *r, __m256i *y1, __m256i *y2)
{
	uint64_t sample[8] __attribute__ ((aligned (32)));
	uint32_t i = 0, j = 0;
	uint64_t x;
	
	while (j < 8)
	{
		do
		{	/* we ignore the low probability of rerunning the PRG 
			 * change the loading size i.e. uint16_t according to your UNIFORM_SIZE */
			x = *((uint8_t *)(r + UNIFORM_SIZE * (i++)));
		} while (1 ^ ((x - UNIFORM_Q) >> 63));

		x = x - ((((x * BARRETT_FACTOR) >> BARRETT_BITSHIFT) + 1) * BINARY_SAMPLER_K);
		x = x + (x >> 63) * BINARY_SAMPLER_K;
		
		sample[j++] = x;
	}
	
	*y1 = _mm256_load_si256((__m256i *)(sample));
	*y2 = _mm256_load_si256((__m256i *)(sample + 4));
}

/* FACCT binary sampling algorithm 
 * we compute 8 samples every time by using the AVX2, 
 * then do the rejection */
void sample_0z(POLY_64 *sample)
{
	__m256i v_x, v_y1, v_y2, v_z, v_b_in;
	uint64_t z[8] __attribute__ ((aligned (32)));
	uint64_t b[8] __attribute__ ((aligned (32)));
	
	unsigned char r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1] __attribute__ ((aligned (32)));
	unsigned char *r1;
	
	uint32_t i = 8, j = 0;
	uint64_t k;
	
	while (j < N)
	{
		do
		{
			if (i == 8)
			{
				/* x<--D_sigma_0, y<--U([0,k-1]), z=kx+y */
				fastrandombytes(r, 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1);
				
				uniform_sampler(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), &v_y1, &v_y2);
				
				r1 = r;
				v_x = cdt_sampler(r1);
				v_x = _mm256_mul_epu32(v_x, V_K_K_K_K);
				v_z = _mm256_add_epi64(v_x, v_y1);
				_mm256_store_si256((__m256i *)(z), v_z);
				/* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				v_b_in = _mm256_add_epi64(v_z, v_x);
				v_b_in = _mm256_mul_epu32(v_b_in, v_y1);
				bernoulli_sampler(b, v_b_in, r1 + BASE_TABLE_SIZE);
				
				r1 = r + (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE);
				v_x = cdt_sampler(r1);
				v_x = _mm256_mul_epu32(v_x, V_K_K_K_K);
				v_z = _mm256_add_epi64(v_x, v_y2);
				_mm256_store_si256((__m256i *)(z + 4), v_z);
				/* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				v_b_in = _mm256_add_epi64(v_z, v_x);
				v_b_in = _mm256_mul_epu32(v_b_in, v_y2);
				bernoulli_sampler(b + 4, v_b_in, r1 + BASE_TABLE_SIZE);

				i = 0;
			}
			
			k = (r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE] >> i) & 0x1;
			i++;			
		} while (1 ^ ((b[i - 1] & ((z[i - 1] | -z[i - 1]) | (k | -k))) >> 63)); /* rejection condition: b=0 or ((b=1) && (z=0) && (k=0)) */
		
		sample->poly[j++] = z[i - 1] * (1 ^ ((-k) & 0xfffffffffffffffe)); /* sample=z*(-1)^k */
	}
}
