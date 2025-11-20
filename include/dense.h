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
    Matrix* weight_grad;
    Matrix* bias_grad;
    
    // auxiliary gradients
    Matrix* dL_dA; 
    Matrix* dA_dZ;

    // transpose buffers
    Matrix* weight_t;
    Matrix* input_t;
} DenseCache;