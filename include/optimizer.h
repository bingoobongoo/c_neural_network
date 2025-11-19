#pragma once

#include "config.h"
#include "matrix.h"
#include "scalar.h"
#include "tensor.h"

typedef struct Optimizer Optimizer;

typedef enum {
    SGD,
    MOMENTUM,
    NESTEROV,
    ADAGRAD,
    ADAM
} OptimizerType;    

struct Optimizer {
    void (*update_conv_weights)(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx);
    void (*update_dense_weights)(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
    void (*update_bias)(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);
    void (*update_batch_norm_gamma)(
        Matrix* gamma, 
        Matrix* gamma_gradient, 
        Optimizer* optimizer, 
        int layer_idx
    );
    void (*update_batch_norm_beta)(
        Matrix* beta,
        Matrix* beta_grad,
        Optimizer* optimizer,
        int layer_idx
    );
    void (*optimizer_free)(Optimizer* optimizer);
    void (*optimizer_print_info)(Optimizer* optimizer);
    size_t (*optimizer_get_mem_allocated)(Optimizer* optimizer);
    OptimizerType type;
    char* name;
    void* settings;
};

typedef struct {
    nn_float learning_rate;
} SGDConfig;

typedef struct {
    nn_float learning_rate;
    nn_float beta;
    int n_layers;
    Tensor4D** weight_momentum;
    Matrix** bias_momentum;
} MomentumConfig;

typedef struct {
    nn_float learning_rate;
    int n_layers;
    Tensor4D** weight_s;
    Matrix** bias_s;
    Tensor4D** intermediate_w;
    Matrix** intermediate_b;
} AdaGradConfig;

typedef struct {
    nn_float learning_rate;
    nn_float beta_m;
    nn_float beta_s;
    int n_layers;
    int ctr;

    Tensor4D** weight_m;
    Tensor4D** weight_m_corr;
    Tensor4D** weight_s;
    Tensor4D** weight_s_corr;
    Tensor4D** intermediate_w;
    
    Matrix** bias_m;
    Matrix** bias_m_corr;
    Matrix** bias_s;
    Matrix** bias_s_corr;
    Matrix** intermediate_b;
} AdamConfig;

Optimizer* optimizer_sgd_new(nn_float learning_rate);
void optimizer_sgd_free(Optimizer* optimizer);
void update_dense_weights_sgd(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_conv_weights_sgd(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_sgd(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void optimizer_sgd_print_info(Optimizer* optimizer);
size_t optimizer_sgd_get_mem_allocated(Optimizer* optimizer);

Optimizer* optimizer_momentum_new(nn_float learning_rate, nn_float beta, bool nesterov);
void optimizer_momentum_free(Optimizer* optimizer);
void update_dense_weights_momentum(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_conv_weights_momentum(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_momentum(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void pre_update_dense_weights_nesterov(Matrix* weight, Optimizer* optimizer, int layer_idx);
void pre_update_conv_weights_nesterov(Tensor4D* weight, Optimizer* optimizer, int layer_idx);
void pre_update_bias_nesterov(Matrix* bias, Optimizer* optimizer, int layer_idx);
void optimizer_momentum_print_info(Optimizer* optimizer);
size_t optimizer_momentum_get_mem_allocated(Optimizer* optimizer);

Optimizer* optimizer_adagrad_new(nn_float learning_rate);
void optimizer_adagrad_free(Optimizer* optimizer);
void update_dense_weights_adagrad(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_conv_weights_adagrad(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_adagrad(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void optimizer_adagrad_print_info(Optimizer* optimizer);
size_t optimizer_adagrad_get_mem_allocated(Optimizer* optimizer);

Optimizer* optimizer_adam_new(nn_float learning_rate, nn_float beta_m, nn_float beta_s);
void optimizer_adam_free(Optimizer* optimizer);
void update_dense_weights_adam(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void update_conv_weights_adam(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx);
void update_bias_adam(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx);
void optimizer_adam_print_info(Optimizer* optimizer);
size_t optimizer_adam_get_mem_allocated(Optimizer* optimizer);