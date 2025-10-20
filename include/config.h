#pragma once

// CONSTANTS

#define PI 3.14159265358979323846

// =============================================================================

// FUNCTIONALITIES

/*
Enables debug informations for basic linear algebra operations, such us dimension
checks, out of bounds checks etc. Very slow, use only when debugging.
*/
// #define DEBUG

/*
Enables optimized library for basic linear algerba operations, effectively 
replacing custom operations with their optimized counterparts from CBLAS library.
*/
#define BLAS

/*
Enables im2col method for calculating convolution for forward and backward passes in CNN.
*/
// #define IM2COL_CONV

// =============================================================================

// TYPE ALIASES

/*
Defined if using single precision (32-bit) float as basic primitive type.
*/
#define SINGLE_PRECISION

#ifndef SINGLE_PRECISION
/*
Defined if using double precision (64-bit) float as basic primitive type.
*/
#define DOUBLE_PRECISION
#endif

#ifdef SINGLE_PRECISION
/*
Alias to quickly change basic primitive type with which all the calculations are
performed. The default is 32-bit floating point (single precision). 
*/
#define nn_float float
#endif

#ifdef DOUBLE_PRECISION
#define nn_float double
#endif


// =============================================================================

// OPTIMIZATIONS

// #define INLINE

// =============================================================================