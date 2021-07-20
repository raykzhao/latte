/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * NTT                            *
 * ****************************** */
 
#ifndef _NTT_H
#define _NTT_H

#include "poly.h"

void ntt_core(int64_t *a, const uint64_t *root, const uint64_t q, const uint64_t montgomery_factor, const uint64_t montgomery_convert_factor);
void ntt(POLY_64 *a);

void intt_core(int64_t *a, const uint64_t *root_inv, const uint64_t q, const uint64_t montgomery_factor, const uint64_t inv_n);
void intt(POLY_64 *a);


#endif
