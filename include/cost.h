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
    double (*cost_func)(double, double);
    double (*dA)(double, double);
} Cost;

Cost* cost_new(CostType type);
void cost_free(Cost* cost);

Matrix* apply_cost_func(Cost* cost, Matrix* output_activation_m, Matrix* label_m);
void apply_cost_func_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into);
Matrix* apply_cost_dA(Cost* cost, Matrix* output_activation_m, Matrix* label_m);
void apply_cost_dA_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into);
double get_avg_batch_loss(Cost* cost, Matrix* output_activation_m, Matrix* label_m);

double mse(double output_activation, double label);
double mse_dA(double output_activation, double label);

double cat_cross_entropy(double output_activation, double label);
double cat_cross_entropy_dA(double output_activation, double label);
