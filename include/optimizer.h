#pragma once

#include "matrix.h"

typedef struct Optimizer Optimizer;

typedef enum {
    SGD,
    MOMENTUM,
    NESTEROV,
    ADAGRAD,
    ADAM
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

typedef struct {
    double learning_rate;
    int n_layers;
    Matrix** weight_s;
    Matrix** bias_s;
    Matrix** intermediate_w;
    Matrix** intermediate_b;
} AdaGradConfig;

typedef struct {
    double learning_rate;
    double beta_m;
    double beta_s;
    int n_layers;
    int ctr;

    Matrix** weight_m;
    Matrix** weight_m_corr;
    Matrix** weight_s;
    Matrix** weight_s_corr;
    Matrix** intermediate_w;
    
    Matrix** bias_m;
    Matrix** bias_m_corr;
    Matrix** bias_s;
    Matrix** bias_s_corr;
    Matrix** intermediate_b;
} AdamConfig;

Optimizer* optimizer_sgd_new(double learning_rate);
void optimizer_sgd_free(Optimizer* optimizer);
void update_weights_sgd(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_sgd(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);

Optimizer* optimizer_momentum_new(double learning_rate, double beta, bool nesterov);
void optimizer_momentum_free(Optimizer* optimizer);
void update_weights_momentum(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_momentum(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);

Optimizer* optimizer_adagrad_new(double learning_rate);
void optimizer_adagrad_free(Optimizer* optimizer);
void update_weights_adagrad(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_adagrad(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);

Optimizer* optimizer_adam_new(double learning_rate, double beta_m, double beta_s);
void optimizer_adam_free(Optimizer* optimizer);
void update_weights_adam(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_adam(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);