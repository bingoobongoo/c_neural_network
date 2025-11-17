#pragma once

#include "config.h"
#include "matrix.h"
#include "batch.h"

typedef enum {
    SIGMOID,
    RELU,
    LRELU,
    ELU,
    SOFTMAX,
    IDENTITY
} ActivationType;

typedef struct {
    ActivationType type;
    Batch* y_true_batch;
    nn_float(*activation_func)(nn_float, nn_float);
    nn_float(*dZ)(nn_float, nn_float);
    nn_float activation_param;
    char* name;
} Activation;

Activation* activation_new(ActivationType type, nn_float param);

Matrix* apply_activation_func(Activation* activation, Matrix* z_m);
void apply_activation_func_into(Activation* activation, Matrix* z_m, Matrix* into);
Matrix* apply_activation_dZ(Activation* activation, Matrix* z_m);
void apply_activation_dZ_into(Activation* activation, Matrix* z_m, Matrix* into);

nn_float sigmoid(nn_float z, nn_float param);
nn_float sigmoid_dZ(nn_float z, nn_float param);

nn_float relu(nn_float z, nn_float param);
nn_float relu_dZ(nn_float z, nn_float param);

nn_float leaky_relu(nn_float z, nn_float param);
nn_float leaky_relu_dZ(nn_float z, nn_float param);

nn_float elu(nn_float z, nn_float param);
nn_float elu_dZ(nn_float z, nn_float param);

nn_float identity(nn_float z, nn_float param);
nn_float identity_dZ(nn_float z, nn_float param);