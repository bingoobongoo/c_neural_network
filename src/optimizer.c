#include "optimizer.h"

Optimizer* optimizer_sgd_new(double learning_rate) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_weights = update_weights_sgd;
    opt->update_bias = update_bias_sgd;
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

void update_weights_sgd(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    SGDConfig* sgd = (SGDConfig*)optimizer->settings;
    matrix_scale_inplace(sgd->learning_rate, gradient);
    matrix_subtract_into(weights, gradient, weights);
}

void update_bias_sgd(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    SGDConfig* sgd = (SGDConfig*)optimizer->settings;
    matrix_scale_inplace(sgd->learning_rate, gradient);
    matrix_subtract_into(bias, gradient, bias);
}

Optimizer* optimizer_momentum_new(double learning_rate, double beta, bool nesterov) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_weights = update_weights_momentum;
    opt->update_bias = update_bias_momentum;
    opt->optimizer_free = optimizer_momentum_free;
    if (nesterov) {
        opt->type = NESTEROV;
        opt->name = "Nesterov Accelerated Gradient (NAG)";
    }
    else {
        opt->type = MOMENTUM;
        opt->name = "Momentum";
    }

    MomentumConfig* mom = (MomentumConfig*)malloc(sizeof(MomentumConfig));
    mom->learning_rate = learning_rate;
    mom->beta = beta;
    mom->n_layers = -1;
    mom->weight_momentum = NULL;
    mom->bias_momentum = NULL;

    opt->settings = mom;

    return opt;
}

void optimizer_momentum_free(Optimizer* optimizer) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;

    for (int i=0; i<mom->n_layers; i++) {
        matrix_free(mom->weight_momentum[i]);
        matrix_free(mom->bias_momentum[i]);
    }

    free(mom->weight_momentum);
    free(mom->bias_momentum);
    free(optimizer->settings);
    free(optimizer);
}

void update_weights_momentum(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;

    Matrix* weight_momentum_mat = mom->weight_momentum[layer_idx];
    matrix_scale_inplace(mom->beta, weight_momentum_mat);
    matrix_scale_inplace(mom->learning_rate, gradient);
    matrix_subtract_into(weight_momentum_mat, gradient, weight_momentum_mat);
    matrix_add_into(weights, weight_momentum_mat, weights);
}

void update_bias_momentum(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;

    Matrix* bias_momentum_mat = mom->bias_momentum[layer_idx];
    matrix_scale_inplace(mom->beta, bias_momentum_mat);
    matrix_scale_inplace(mom->learning_rate, gradient);
    matrix_subtract_into(bias_momentum_mat, gradient, bias_momentum_mat);
    matrix_add_into(bias, bias_momentum_mat, bias);
}

Optimizer* optimizer_adagrad_new(double learning_rate) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_weights = update_weights_adagrad;
    opt->update_bias = update_bias_adagrad;
    opt->optimizer_free = optimizer_adagrad_free;
    opt->type = ADAGRAD;
    opt->name = "AdaGrad";

    AdaGradConfig* ada = (AdaGradConfig*)malloc(sizeof(AdaGradConfig));
    ada->learning_rate = learning_rate;
    ada->n_layers = -1;
    ada->weight_s = NULL;
    ada->bias_s = NULL;
    ada->intermediate_w = NULL;
    ada->intermediate_b = NULL;

    opt->settings = ada;

    return opt;
}

void optimizer_adagrad_free(Optimizer* optimizer) {
    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;

    for (int i=0; i<ada->n_layers; i++) {
        matrix_free(ada->weight_s[i]);
        matrix_free(ada->bias_s[i]);
        matrix_free(ada->intermediate_w[i]);
        matrix_free(ada->intermediate_b[i]);
    }

    free(ada->weight_s);
    free(ada->bias_s);
    free(ada->intermediate_w);
    free(ada->intermediate_b);
    free(optimizer->settings);
    free(optimizer);
}

void update_weights_adagrad(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;

    Matrix* weight_s = ada->weight_s[layer_idx];
    Matrix* intermediate = ada->intermediate_w[layer_idx];
    matrix_multiply_into(gradient, gradient, intermediate);
    matrix_add_into(weight_s, intermediate, weight_s);
    
    matrix_scale_inplace(ada->learning_rate, gradient);
    matrix_add_scalar_into(1e-10, weight_s, intermediate);
    matrix_apply_inplace(sqrt, intermediate);
    matrix_divide_into(gradient, intermediate, intermediate);
    matrix_subtract_into(weights, intermediate, weights);
}

void update_bias_adagrad(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;

    Matrix* bias_s = ada->bias_s[layer_idx];
    Matrix* intermediate = ada->intermediate_b[layer_idx];
    matrix_multiply_into(gradient, gradient, intermediate);
    matrix_add_into(bias_s, intermediate, bias_s);
    
    matrix_scale_inplace(ada->learning_rate, gradient);
    matrix_add_scalar_into(1e-10, bias_s, intermediate);
    matrix_apply_inplace(sqrt, intermediate);
    matrix_divide_into(gradient, intermediate, intermediate);
    matrix_subtract_into(bias, intermediate, bias);
}