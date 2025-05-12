#pragma once

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
    float (*cost_func)(float, float);
    float (*dA)(float, float);
} Cost;

Cost* cost_new(CostType type);
void cost_free(Cost* cost);

Matrix* apply_cost_func(Cost* cost, Matrix* output_activation_m, Matrix* label_m);
void apply_cost_func_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into);
Matrix* apply_cost_dA(Cost* cost, Matrix* output_activation_m, Matrix* label_m);
void apply_cost_dA_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into);
float get_avg_batch_loss(Cost* cost, Matrix* output_activation_m, Matrix* label_m);

float mse(float output_activation, float label);
float mse_dA(float output_activation, float label);

float cat_cross_entropy(float output_activation, float label);
float cat_cross_entropy_dA(float output_activation, float label);
