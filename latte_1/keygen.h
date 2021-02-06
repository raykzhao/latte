/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Keygen                         *
 * ****************************** */
 
#ifndef _KEYGEN_H
#define _KEYGEN_H

#include "poly.h"
#include "mat.h"

void keygen(MAT_64 *basis, POLY_64 *h, POLY_64 *b, const unsigned char *seed);

#endif
