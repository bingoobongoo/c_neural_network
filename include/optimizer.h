#pragma once

#include "matrix.h"

typedef struct Optimizer Optimizer;

typedef enum {
    SGD,
    MOMENTUM,
    NESTEROV
} OptimizerType;    

struct Optimizer {
    void (*update_weights)(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
    void (*update_bias)(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);
    void (*optimizer_free)(Optimizer* optimizer);
    OptimizerType type;
    char* name;
    void* settings;
};

typedef struct {
    double learning_rate;
} SGDConfig;

typedef struct {
    double learning_rate;
    double beta;
    int n_layers;
    Matrix** weight_momentum;
    Matrix** bias_momentum;
} MomentumConfig;

Optimizer* optimizer_sgd_new(double learning_rate);
void optimizer_sgd_free(Optimizer* optimizer);
void update_weights_sgd(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_sgd(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);

Optimizer* optimizer_momentum_new(double learning_rate, double beta, bool nesterov);
void optimizer_momentum_free(Optimizer* optimizer);
void update_weights_momentum(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_momentum(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);
