#pragma once

#include "config.h"
#include "matrix.h"

typedef enum {
    MSE,
    CAT_CROSS_ENTROPY,
    BIN_CROSS_ENTROPY
} LossType;

typedef struct {
    LossType loss_type;
    char* name;
    Matrix* loss_m;
    nn_float (*loss_func)(nn_float, nn_float);
    nn_float (*dA)(nn_float, nn_float);
} Loss;

Loss* loss_new(LossType type);
void loss_free(Loss* loss);

Matrix* apply_loss_func(Loss* loss, Matrix* output_activation_m, Matrix* label_m);
void apply_loss_func_into(Loss* loss, Matrix* output_activation_m, Matrix* label_m, Matrix* into);
Matrix* apply_loss_dA(Loss* loss, Matrix* output_activation_m, Matrix* label_m);
void apply_loss_dA_into(Loss* loss, Matrix* output_activation_m, Matrix* label_m, Matrix* into);
nn_float get_avg_batch_loss(Loss* loss, Matrix* output_activation_m, Matrix* label_m);

nn_float mse(nn_float output_activation, nn_float label);
nn_float mse_dA(nn_float output_activation, nn_float label);

nn_float cat_cross_entropy(nn_float output_activation, nn_float label);
nn_float cat_cross_entropy_dA(nn_float output_activation, nn_float label);
