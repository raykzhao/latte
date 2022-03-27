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

#endif
