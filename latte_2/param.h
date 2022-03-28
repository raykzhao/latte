#ifndef _PARAM_H
#define _PARAM_H

#include <stdint.h>

#define N 2048

#define Q 33550337LL /* 2^25 - 2^12 + 1 */

#define L 1

static const double sigma_l[L + 1] = {106.165225036002, 7900.24361166743}; 

static const double norm_l[L] = {46166241.3084468};

#define REDUCE_K_PREC 384

#endif
