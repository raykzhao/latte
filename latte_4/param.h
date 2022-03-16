#ifndef _PARAM_H
#define _PARAM_H

#include <stdint.h>

#define PREC 88

#define N 2048

#define Q 274810798081LL /* 2^38 - 2^26 + 1 */

#define L 2

static const char sigma_str[L + 1][PREC] = {"9583.7471944500068780694380", "713170.79131244259284534655", "65489528.122674807214409517"}; 

static const char norm_str[L][PREC] = {"376210269336.07783606747078", "3124915676658988.2977945632"};

static const uint64_t reduce_k_prec[L] = {512, 1536};
#define FFT_PREC 1536

#endif
