#pragma once

#include "matrix.h"
#include "tensor.h"

typedef struct {
    int n_filters;
    int filter_size;
    int stride;
    int n_units;
} ConvParams;

typedef struct {
    Tensor4D* output;
    Tensor4D* z;
    Tensor4D* filter;
    Tensor4D* bias;
    Tensor4D* delta;
    Tensor4D* filter_gradient;
    Tensor4D* bias_gradient;

    // auxiliary
    Tensor4D* dCost_dA;
    Tensor4D* dActivation_dZ;
    Matrix* fp_im2col_input;
    Matrix* fp_im2col_kernel;
    Matrix* fp_im2col_output;
    Matrix* dCost_dW_im2col_input;
    Matrix* dCost_dW_im2col_kernel;
    Matrix* dCost_dW_im2col_output;
    Matrix* delta_im2col_input;
    Matrix* delta_im2col_kernel;
    Matrix* delta_im2col_output;
} ConvCache;