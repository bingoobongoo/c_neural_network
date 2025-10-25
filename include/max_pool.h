#pragma once

#include "config.h"
#include "matrix.h"
#include "tensor.h"

typedef struct {
    int filter_size;
    int stride;
    int n_units;
    int output_width;
    int output_height;
} MaxPoolParams;

typedef struct {
    Tensor4D* output;
    Tensor4D* delta;
    Tensor4D* dCost_dA;
    Tensor4D_uint16* argmax;
} MaxPoolCache;