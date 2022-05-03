#ifndef _RED_H
#define _RED_H

#include <stdint.h>
#include "param.h"

/* x - q if x >= q */
static inline uint64_t con_sub(const uint64_t x, const uint64_t q)
{
	return x - ((-(1 ^ ((x - q) >> 63))) & q);
}

/* x + q if x <= q */
static inline uint64_t con_add(const uint64_t x, const uint64_t q)
{
	return x + ((-(x >> 63)) & q);
}

#define MONTGOMERY_FACTOR 16773119
#define MONTGOMERY_SHIFT 32
#define MONTGOMERY_CONVERT_FACTOR 33546244
#define MONTGOMERY_INV_FACTOR 1834688

/* Montgomery reduction
 * Input: x < Q*R, where R=2^k and Q<R
 * Output: m = x*R^{-1} % Q
 * 
 * b = -Q^{-1} % R
 * t = ((x % R)*b) % R
 * m = (x + t * Q) / R */
static inline uint64_t montgomery(uint64_t a, uint64_t b)
{
	uint64_t t;
	uint32_t x, y;
	
	t = a * b;
	x = t;
	y = ((uint64_t)x) * MONTGOMERY_FACTOR;
	
	return con_sub((t + ((uint64_t)y) * Q) >> MONTGOMERY_SHIFT, Q);
}

/* Modular inverse mod q */
static inline int64_t inverse(int64_t a)
{
	int64_t t = 0;
	int64_t newt = 1;
	int64_t r = Q;
	int64_t newr = a;
	int64_t q;
	int64_t tmp;

	while (newr)
	{
		q = r / newr;
		tmp = newt;
		newt = t - q * newt;
		t = tmp;
		tmp = newr;
		newr = r - q * newr;
		r = tmp;
	}
	
	return con_add(t, Q);
}

/* Barrett reduction
 * Input: x < 2^k
 * Output m = x % Q in [0, 2Q)
 * 
 * b = floor(2^k/Q)
 * t = floor((x * b) / 2^k), where t is an estimation of x / Q
 * m = x - t * Q */
static inline uint64_t barrett(const uint64_t x, const uint64_t barrett_factor, const uint64_t barrett_shift)
{
	return con_sub(x - ((x * barrett_factor) >> barrett_shift) * Q, Q);
}

#endif
