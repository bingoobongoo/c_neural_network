#include "cost.h"

Cost* cost_new(CostType type) {
    Cost* cost = (Cost*)malloc(sizeof(Cost));
    cost->cost_type = type;
    cost->loss_m = NULL;
    switch (type)
    {
    case MSE:
        cost->cost_func = mse;
        cost->dA = mse_dA;
        cost->name = "Mean Squared Error (MSE)";
        break;
    
    case CAT_CROSS_ENTROPY:
        cost->cost_func = cat_cross_entropy;
        cost->dA = cat_cross_entropy_dA;
        cost->name = "Categorical Cross-Entropy";
        break;
    }

    return cost;
}

void cost_free(Cost* cost) {
    matrix_free(cost->loss_m);
    free(cost);
}

Matrix* apply_cost_func(Cost* cost, Matrix* output_activation_m, Matrix* label_m) {
    Matrix* err_m = matrix_new(output_activation_m->n_rows, output_activation_m->n_cols);
    for (int i=0; i<output_activation_m->n_rows; i++) {
        for (int j=0; j<output_activation_m->n_cols; j++) {
            nn_float output_activation = matrix_get(output_activation_m, i, j);
            nn_float label = matrix_get(label_m, i, j);
            matrix_assign(err_m, i, j, cost->cost_func(output_activation, label));
        }
    }

    return err_m;
}

void apply_cost_func_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into) {
    for (int i=0; i<output_activation_m->n_rows; i++) {
        for (int j=0; j<output_activation_m->n_cols; j++) {
            nn_float output_activation = matrix_get(output_activation_m, i, j);
            nn_float label = matrix_get(label_m, i, j);
            matrix_assign(into, i, j, cost->cost_func(output_activation, label));
        }
    }
}

Matrix* apply_cost_dA(Cost* cost, Matrix* output_activation_m, Matrix* label_m) {
    Matrix* dA = matrix_new(output_activation_m->n_rows, output_activation_m->n_cols);
    for (int i=0; i<output_activation_m->n_rows; i++) {
        for (int j=0; j<output_activation_m->n_cols; j++) {
            nn_float output_activation = matrix_get(output_activation_m, i, j);
            nn_float label = matrix_get(label_m, i, j);
            matrix_assign(dA, i, j, cost->dA(output_activation, label));
        }
    }

    return dA;
}

void apply_cost_dA_into(Cost* cost, Matrix* output_activation_m, Matrix* label_m, Matrix* into) {
    for (int i=0; i<output_activation_m->n_rows; i++) {
        for (int j=0; j<output_activation_m->n_cols; j++) {
            nn_float output_activation = matrix_get(output_activation_m, i, j);
            nn_float label = matrix_get(label_m, i, j);
            matrix_assign(into, i, j, cost->dA(output_activation, label));
        }
    }
}

nn_float get_avg_batch_loss(Cost* cost, Matrix* output_activation_m, Matrix* label_m) {
    nn_float avg_loss = 0.0;
    apply_cost_func_into(cost, output_activation_m, label_m, cost->loss_m);
    avg_loss = matrix_sum(cost->loss_m) / (label_m->n_rows * label_m->n_cols);

    return avg_loss;

}

nn_float mse(nn_float output_activation, nn_float label) {
    return pow(output_activation - label, 2.0);
}

nn_float mse_dA(nn_float output_activation, nn_float label) {
    return 2.0 * (output_activation - label);
}

nn_float cat_cross_entropy(nn_float output_activation, nn_float label) {
    if (label > 0.0)
        return -label * log(output_activation + 1e-9);
    return 0.0;
}

nn_float cat_cross_entropy_dA(nn_float output_activation, nn_float label) {
    // simplified version of derivative, dC/dZ = y_pred-y_true with softmax, so
    // because dC/dZ = dA/dZ * dC/dA and dA/dZ = y_pred-y_true, then
    // dC/dA = 1
    return 1.0;
}   