#pragma once

#include <immintrin.h>

#include "config.h"

#ifdef SINGLE_PRECISION
    typedef __m256 simd_vec;
    #define NN_SIMD_WIDTH 8

    #define SIMD_LOAD(p)      _mm256_loadu_ps(p)
    #define SIMD_STORE(p,v)   _mm256_storeu_ps((p),(v))
    #define SIMD_SET1(x)      _mm256_set1_ps((x))
    #define SIMD_ADD(a,b)     _mm256_add_ps((a),(b))
    #define SIMD_SUB(a,b)     _mm256_sub_ps((a),(b))
    #define SIMD_MUL(a,b)     _mm256_mul_ps((a),(b))
    #define SIMD_DIV(a,b)     _mm256_div_ps((a),(b))

#elif defined(DOUBLE_PRECISION)
    typedef __m256d simd_vec;
    #define NN_SIMD_WIDTH 4

    #define SIMD_LOAD(p)      _mm256_loadu_pd(p)
    #define SIMD_STORE(p,v)   _mm256_storeu_pd((p),(v))
    #define SIMD_SET1(x)      _mm256_set1_pd((x))
    #define SIMD_ADD(a,b)     _mm256_add_pd((a),(b))
    #define SIMD_SUB(a,b)     _mm256_sub_pd((a),(b))
    #define SIMD_MUL(a,b)     _mm256_mul_pd((a),(b))
    #define SIMD_DIV(a,b)     _mm256_div_pd((a),(b))
#endif

void simd_add(
    const nn_float* a,
    const nn_float* b,
    nn_float* c,
    int n
);

void simd_sub(
    const nn_float* a,
    const nn_float* b,
    nn_float* c,
    int n
); 

void simd_mul(
    const nn_float* a,
    const nn_float* b,
    nn_float* c,
    int n
);

void simd_div(
    const nn_float* a,
    const nn_float* b,
    nn_float* c,
    int n
);

void simd_scale(
    const nn_float* a,
    nn_float s,
    nn_float* c,
    int n
);

void simd_add_scalar(
    const nn_float* a,
    nn_float s,
    nn_float* c,
    int n
);

nn_float simd_sum(
    const nn_float* a,
    int n
);