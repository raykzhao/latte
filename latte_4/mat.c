/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Matrice arithmetic             *
 * ****************************** */
 
#include <stdint.h>
#include "mat.h"
#include "param.h"
#include "poly.h"

#include <mpc.h>
#include <mpfr.h>

/* Compute the Gram matrix a * (a*)^T */
void gram(MAT_FFT *out, const MAT_FFT *a, const uint64_t dim, const uint64_t n)
{
	uint64_t i, j, k, p;
	
	static MAT_FFT a_head;
	
	mpfr_t tmp, tmp1;
	mpfr_inits2(PREC, tmp, tmp1, NULL);
	
	mat_fft_init(&a_head, dim, n);
	
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			for (p = 0; p < N; p++)
			{
				mpc_conj(a_head.mat[i][j].poly[p], a->mat[i][j].poly[p], MPC_RNDNN);
			}
		}
	}
	
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			if (i == j)
			{
				for (p = 0; p < N; p++)
				{
					mpc_norm(tmp, a->mat[i][0].poly[p], MPFR_RNDN);
					for (k = 1; k < dim; k++)
					{
						mpc_norm(tmp1, a->mat[i][k].poly[p], MPFR_RNDN);
						mpfr_add(tmp, tmp, tmp1, MPFR_RNDN);
					}
					mpc_set_fr(out->mat[i][j].poly[p], tmp, MPC_RNDNN);
				}
			}
			else
			{
				for (p = 0; p < N; p++)
				{
					mpc_mul(out->mat[i][j].poly[p], a->mat[i][0].poly[p], a_head.mat[j][0].poly[p], MPC_RNDNN);
					for (k = 1; k < dim; k++)
					{
						mpc_fma(out->mat[i][j].poly[p], a->mat[i][k].poly[p], a_head.mat[j][k].poly[p], out->mat[i][j].poly[p], MPC_RNDNN);
					}
				}
			}
		}
	}
	
	mat_fft_clear(&a_head, dim, n);
	
	mpfr_clears(tmp, tmp1, NULL);
}
