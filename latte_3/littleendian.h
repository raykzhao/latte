/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * bytes <--> integers            *
 * ****************************** */

#ifndef _LITTLEENDIAN_H
#define _LITTLEENDIAN_H

#include <stdint.h>

static inline uint64_t load_40(const unsigned char *x)
{
	return ((uint64_t)(*x)) | (((uint64_t)(*(x + 1))) << 8) | (((uint64_t)(*(x + 2))) << 16) | (((uint64_t)(*(x + 3))) << 24) | (((uint64_t)(*(x + 4))) << 32);
}

static inline __uint128_t load_96(const unsigned char *x)
{
	return ((__uint128_t)(*x)) | (((__uint128_t)(*(x + 1))) << 8) | (((__uint128_t)(*(x + 2))) << 16) | (((__uint128_t)(*(x + 3))) << 24) | (((__uint128_t)(*(x + 4))) << 32) | (((__uint128_t)(*(x + 5))) << 40) | (((__uint128_t)(*(x + 6))) << 48) | (((__uint128_t)(*(x + 7))) << 56) | (((__uint128_t)(*(x + 8))) << 64) | (((__uint128_t)(*(x + 9))) << 72) | (((__uint128_t)(*(x + 10))) << 80) | (((__uint128_t)(*(x + 11))) << 88);
}

#endif
