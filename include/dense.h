#pragma once

#include "config.h"
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
} DenseCache;