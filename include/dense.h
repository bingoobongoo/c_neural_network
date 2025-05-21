#pragma once

#include "matrix.h"
#include "activation.h"

typedef struct {
    int n_units;
} DenseParams;

typedef struct {
    Matrix* output;
    Matrix* z;
    Matrix* weight;
    Matrix* bias;
    Matrix* delta;
    Matrix* weight_gradient;
    Matrix* bias_gradient;
    
    // auxiliary gradients
    Matrix* dCost_dA; 
    Matrix* dActivation_dZ;
    Matrix* dZ_dW_t;
    Matrix* dZnext_dA_t;
    Matrix* dCost_dZ_col_sum;
} DenseCache;