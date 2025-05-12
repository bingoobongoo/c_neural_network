#pragma once

#include "matrix.h"
#include "batch.h"

typedef enum {
    SIGMOID,
    RELU,
    LRELU,
    ELU,
    SOFTMAX,
} ActivationType;

typedef struct {
    ActivationType type;
    Batch* y_true_batch;
    float(*activation_func)(float, float);
    float(*dZ)(float, float);
    float activation_param;
    char* name;
} Activation;

Activation* activation_new(ActivationType type, float param);

Matrix* apply_activation_func(Activation* activation, Matrix* z_m);
void apply_activation_func_into(Activation* activation, Matrix* z_m, Matrix* into);
Matrix* apply_activation_dZ(Activation* activation, Matrix* z_m);
void apply_activation_dZ_into(Activation* activation, Matrix* z_m, Matrix* into);

float sigmoid(float z, float param);
float sigmoid_dZ(float z, float param);

float relu(float z, float param);
float relu_dZ(float z, float param);

float leaky_relu(float z, float param);
float leaky_relu_dZ(float z, float param);

float elu(float z, float param);
float elu_dZ(float z, float param);