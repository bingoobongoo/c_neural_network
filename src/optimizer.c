#include "optimizer.h"

Optimizer* optimizer_new(OptimizerType type, double learining_rate) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    optimizer->type = type;
    optimizer->learning_rate = learining_rate;

    return optimizer;
}

void update_params_inplace(Matrix* params, Matrix* gradient, Optimizer* optimizer) {
    matrix_scale_inplace(optimizer->learning_rate, gradient);
    matrix_subtract_into(params, gradient, params);
}