#include "optimizer.h"

Optimizer* optimizer_sgd_new(double learning_rate) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_params = update_params_sgd;
    opt->optimizer_free = optimizer_sgd_free;
    opt->type = SGD;
    opt->name = "Stochastic Gradient Descent (SGD)";

    SGDConfig* sgd = (SGDConfig*)malloc(sizeof(SGDConfig));
    sgd->learning_rate = learning_rate;

    opt->settings = sgd;

    return opt;
}

void optimizer_sgd_free(Optimizer* optimizer) {
    free(optimizer->settings);
    free(optimizer);
}

void update_params_sgd(Matrix* params, Matrix* gradient, Optimizer* optimizer) {
    SGDConfig* sgd = (SGDConfig*)optimizer->settings;
    matrix_scale_inplace(sgd->learning_rate, gradient);
    matrix_subtract_into(params, gradient, params);
}