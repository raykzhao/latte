/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Encryption                     *
 * ****************************** */
 
#ifndef _ENCRYPT_H
#define _ENCRYPT_H

#include <stdint.h>
#include "poly.h"

void encrypt(unsigned char *z, POLY_64 *c, const unsigned char *mu, const POLY_64 *a, const POLY_64 *b, const uint64_t l, const unsigned char *seed);


#endif
