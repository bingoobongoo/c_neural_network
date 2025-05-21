#pragma once

#include "matrix.h"

typedef struct {
    int n_units;
} FlattenParams;

typedef struct {
    Matrix* output;
    Matrix* dCost_dA_matrix;
    Matrix* dZnext_dA_t;
} FlattenCache;