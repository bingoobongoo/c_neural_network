#pragma once

#include "config.h"
#include "matrix.h"

typedef enum {
    MSE,
    CAT_CROSS_ENTROPY,
    BIN_CROSS_ENTROPY
} CostType;

typedef struct {
    CostType cost_type;
    char* name;
    Matrix* loss_m;
    nn_float (*cost_func)(nn_float, nn_float);
    nn_float (*dA)(nn_float, nn_float);
} Cost;

Cost* cost_new(CostType type);
void cost_free(Cost* cost);

Matrix* apply_cost_func(Cost* cost, Matrix* output_activation_m, Matrix* label_m);
void apply_cost_func_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into);
Matrix* apply_cost_dA(Cost* cost, Matrix* output_activation_m, Matrix* label_m);
void apply_cost_dA_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into);
nn_float get_avg_batch_loss(Cost* cost, Matrix* output_activation_m, Matrix* label_m);

nn_float mse(nn_float output_activation, nn_float label);
nn_float mse_dA(nn_float output_activation, nn_float label);

nn_float cat_cross_entropy(nn_float output_activation, nn_float label);
nn_float cat_cross_entropy_dA(nn_float output_activation, nn_float label);
