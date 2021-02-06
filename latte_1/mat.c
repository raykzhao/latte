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

/* Compute the Gram matrix a * (a*)^T */
void gram(MAT_FFT *out, const MAT_FFT *a, const uint64_t dim, const uint64_t n)
{
	uint64_t i, j, k, p;
	
	static MAT_FFT a_head;
	
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
			for (p = 0; p < N; p++)
			{
				mpc_set_ui(out->mat[i][j].poly[p], 0, MPC_RNDNN);
			}
			
			for (k = 0; k < dim; k++)
			{
				for (p = 0; p < N; p++)
				{
					mpc_fma(out->mat[i][j].poly[p], a->mat[i][k].poly[p], a_head.mat[j][k].poly[p], out->mat[i][j].poly[p], MPC_RNDNN);
				}
			}
		}
	}
	
	mat_fft_clear(&a_head, dim, n);
}
