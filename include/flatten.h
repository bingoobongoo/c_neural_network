#pragma once

#include "config.h"
#include "matrix.h"

typedef struct {
    int n_units;
} FlattenParams;

typedef struct {
    Matrix* output;
    Matrix* delta;
} FlattenCache;