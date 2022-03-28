/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Benchmark                      *
 * ****************************** */
 
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "param.h"
#include "poly.h"
#include "mat.h"
#include "keygen.h"
#include "randombytes.h"
#include "extract.h"
#include "encrypt.h"
#include "decrypt.h"
#include "cpucycles.h"

#define BENCHMARK_ROUND 1000

int main()
{
	uint64_t i, r;
	unsigned char seed[32];
	unsigned char mu[32];
	unsigned char z[32];
	
	uint64_t ret;
	
	static MAT_64 basis;
	static POLY_64 h;
	static POLY_64 b;
	static POLY_64 a[L + 1];
	static POLY_64 t[L + 2];
	static POLY_64 c[L + 2];
	
	long long cycle1, cycle2, cycle3, cycle4, cycle5, cycle6, cycle7;
	
	srand(time(NULL));
	
	fprintf(stderr, "Latte-1 benchmark\n");
	for (r = 0; r < BENCHMARK_ROUND; r++)
	{
		fprintf(stderr, "%lu\n", r);
		
		randombytes(seed, 32);
		
		cycle1 = cpucycles();
		keygen(&basis, &h, &b, seed);
		cycle2 = cpucycles();
		
		for (i = 0; i < N; i++)
		{
			a[0].poly[i] = h.poly[i];
			a[1].poly[i] = rand() % Q;
		}

		randombytes(seed, 32);
		
		cycle3 = cpucycles();
		extract(t, &basis, &b, a + 1, 1, seed);
		cycle4 = cpucycles();
		
		randombytes(mu, 32);
		randombytes(seed, 32);
		
		cycle5 = cpucycles();	
		encrypt(z, c, mu, a, &b, 1, seed);
		cycle6 = cpucycles();
		ret = decrypt(mu, z, c, a, &b, t, 1);
		cycle7 = cpucycles();
		
		printf("%lld,%lld,%lld,%lld,%lu\n", cycle2 - cycle1, cycle4 - cycle3, cycle6 - cycle5, cycle7 - cycle6, ret);
	}
	
	return 0;
}
