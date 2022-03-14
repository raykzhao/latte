#ifndef _PARAM_H
#define _PARAM_H

#include <stdint.h>

#define PREC 88

#define N 1024

#define Q 68718428161LL /* 2^36 - 2^20 + 1 */

#define L 2

static const char sigma_str[L + 1][PREC] = {"6777.5879662983919609175926", "351968.36683734171059914430", "22559988.027951553365675150"}; 

static const char norm_str[L][PREC] = {"94076310816.589358454827978", "380564678412735.11125276429"};

#endif
