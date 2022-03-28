#ifndef _PARAM_H
#define _PARAM_H

#include <stdint.h>

#define N 1024

#define Q 16760833LL /* 2^24 - 2^14 + 1 */

#define L 1

static const double sigma_l[L + 1] = {106.165225036002, 5513.28895421901}; 

static const double norm_l[L] = {23083120.6542234};

#define REDUCE_K_PREC 256

#endif
