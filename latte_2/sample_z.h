/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Integer samplers               *
 * ****************************** */
 
#ifndef _SAMPLE_Z_H
#define _SAMPLE_Z_H

#include <mpfr.h>
#include "poly.h"

int64_t sample_z(const mpfr_t center, const mpfr_t sigma);
void sample_e(POLY_64 *out);
void sample_0z(POLY_64 *sample);

#endif
