#pragma once

#include "matrix.h"

typedef enum {
    MSE,
} CostType;

typedef struct {
    CostType cost_type;
    double (*cost_func)(double, double);
    double (*dA)(double, double);
} Cost;

Cost* cost_new(CostType type);

Matrix* apply_cost_func(Cost* cost, Matrix* output_activation_m, Matrix* label_m);
Matrix* apply_cost_dA(Cost* cost, Matrix* output_activation_m, Matrix* label_m);

double mse(double output_activation, double label);
double mse_dA(double output_activation, double label);
