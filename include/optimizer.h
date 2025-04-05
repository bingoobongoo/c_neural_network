#pragma once

#include "matrix.h"

typedef struct Optimizer Optimizer;

typedef enum {
    SGD,
    MOMENTUM
} OptimizerType;    

struct Optimizer {
    void (*update_params)(Matrix* params, Matrix* gradient, Optimizer* optimizer);
    void (*optimizer_free)(Optimizer* optimizer);
    OptimizerType type;
    char* name;
    void* settings;
};

typedef struct {
    double learning_rate;
} SGDConfig;

Optimizer* optimizer_sgd_new(double learning_rate);
void optimizer_sgd_free(Optimizer* optimizer);
void update_params_sgd(Matrix* params, Matrix* gradient, Optimizer* optimizer);
