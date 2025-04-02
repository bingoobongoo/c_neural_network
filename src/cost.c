#include "cost.h"

Cost* cost_new(CostType type) {
    Cost* cost = (Cost*)malloc(sizeof(Cost));
    cost->cost_type = type;
    switch (type)
    {
    case MSE:
        cost->cost_func = mse;
        cost->dA = mse_dA;
        break;
    }

    return cost;
}

Matrix* apply_cost_func(Cost* cost, Matrix* output_activation_m, Matrix* label_m) {
    Matrix* err_m = matrix_new(output_activation_m->n_rows, output_activation_m->n_cols);
    for (int i=0; i<output_activation_m->n_rows; i++) {
        for (int j=0; j<output_activation_m->n_cols; j++) {
            double output_activation = output_activation_m->entries[i][j];
            double label = label_m->entries[i][j];
            err_m->entries[i][j] = cost->cost_func(output_activation, label);
        }
    }

    return err_m;
}

Matrix* apply_cost_dA(Cost* cost, Matrix* output_activation_m, Matrix* label_m) {
    Matrix* dA = matrix_new(output_activation_m->n_rows, output_activation_m->n_cols);
    for (int i=0; i<output_activation_m->n_rows; i++) {
        for (int j=0; j<output_activation_m->n_cols; j++) {
            double output_activation = output_activation_m->entries[i][j];
            double label = label_m->entries[i][j];
            dA->entries[i][j] = cost->dA(output_activation, label);
        }
    }

    return dA;
}

void apply_cost_dA_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into) {
    for (int i=0; i<output_activation_m->n_rows; i++) {
        for (int j=0; j<output_activation_m->n_cols; j++) {
            double output_activation = output_activation_m->entries[i][j];
            double label = label_m->entries[i][j];
            into->entries[i][j] = cost->dA(output_activation, label);
        }
    }
}

double mse(double output_activation, double label) {
    return pow(output_activation - label, 2.0);
}

double mse_dA(double output_activation, double label) {
    return 2.0 * (output_activation - label);
}