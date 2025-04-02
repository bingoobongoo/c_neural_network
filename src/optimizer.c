#include "optimizer.h"

Optimizer* optimizer_new(OptimizerType type, double learining_rate) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    optimizer->type = type;
    optimizer->learning_rate = learining_rate;

    return optimizer;
}

Matrix* update_params(Matrix* params, Matrix* gradient, Optimizer* optimizer) {
    matrix_scale_inplace(optimizer->learning_rate, gradient);
    Matrix* updated_params = matrix_subtract(params, gradient);

    return updated_params;
}