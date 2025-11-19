#pragma once

#include "config.h"
#include "matrix.h"
#include "tensor.h"

typedef struct {
    int n_units;
    nn_float momentum;
} BatchNormDenseParams;

typedef struct {
    Matrix* output;
    Matrix* z;
    Matrix* delta;
    Matrix* dL_dA;
    Matrix* dA_dZ;
    Matrix* x_normalized;

    Matrix* mean;
    Matrix* variance;
    Matrix* running_mean;
    Matrix* running_variance;

    Matrix* gamma;
    Matrix* beta;
    Matrix* gamma_grad;
    Matrix* beta_grad;
} BatchNormDenseCache;

typedef struct {
    int n_units;
    nn_float momentum; // how much new batch stats affect mean and variance (higher = more, def=0.1)
} BatchNormConvParams;

typedef struct {
    Tensor4D* output;
    Tensor4D* z;
    Tensor4D* delta;
    Tensor4D* dL_dA;
    Tensor4D* dA_dZ;
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