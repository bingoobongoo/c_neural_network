#pragma once

#include "matrix.h"

typedef enum {
    SGD
} OptimizerType;    

typedef struct {
    OptimizerType type;
    double learning_rate;
} Optimizer;

Optimizer* optimizer_new(OptimizerType type, double learning_rate);
void update_params_inplace(Matrix* params, Matrix* gradient, Optimizer* optimizer);
