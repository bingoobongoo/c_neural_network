#pragma once

#include "config.h"
#include "matrix.h"
#include "tensor.h"

typedef struct {
    int n_units;
    int output_width;
    int output_height;
    int output_channels;
    int output_filters;
    int momentum;
} BatchNormConvParams;

typedef struct {
    Tensor4D* output;
    Tensor4D* delta;
    Tensor4D* x_normalized;
    Matrix* mean;
    Matrix* variance;
    Matrix* running_mean;
    Matrix* running_variance;
    Matrix* gamma;
    Matrix* beta;
    Matrix* gamma_grad;
    Matrix* beta_grad;
} BatchNormConvCache;