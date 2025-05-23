#include "optimizer.h"

Optimizer* optimizer_sgd_new(nn_float learning_rate) {
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

Optimizer* optimizer_momentum_new(nn_float learning_rate, nn_float beta, bool nesterov) {
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

Optimizer* optimizer_adagrad_new(nn_float learning_rate) {
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
    matrix_add_scalar_into(1e-9, weight_s, intermediate);
    matrix_apply_inplace(sqrtf, intermediate);
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
    matrix_add_scalar_into(1e-9, bias_s, intermediate);
    matrix_apply_inplace(sqrtf, intermediate);
    matrix_divide_into(gradient, intermediate, intermediate);
    matrix_subtract_into(bias, intermediate, bias);
}

Optimizer* optimizer_adam_new(nn_float learning_rate, nn_float beta_m, nn_float beta_s) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_weights = update_weights_adam;
    opt->update_bias = update_bias_adam;
    opt->optimizer_free = optimizer_adam_free;
    opt->type = ADAM;
    opt->name = "Adam";

    AdamConfig* adam = (AdamConfig*)malloc(sizeof(AdamConfig));
    adam->learning_rate = learning_rate;
    adam->beta_m = beta_m;
    adam->beta_s = beta_s;
    adam->n_layers = -1;
    adam->ctr = 1;

    adam->weight_m = NULL;
    adam->bias_m = NULL;
    adam->weight_m_corr = NULL;
    adam->bias_m_corr = NULL;

    adam->weight_s = NULL;
    adam->bias_s = NULL;
    adam->weight_s_corr = NULL;
    adam->bias_m_corr = NULL;

    adam->intermediate_w = NULL;
    adam->intermediate_b = NULL;

    opt->settings = adam;

    return opt;
}

void optimizer_adam_free(Optimizer* optimizer) {
    AdamConfig* adam = (AdamConfig*)optimizer->settings;

    for (int i=0; i<adam->n_layers; i++) {
        matrix_free(adam->weight_m[i]);
        matrix_free(adam->weight_m_corr[i]);
        matrix_free(adam->weight_s[i]);
        matrix_free(adam->weight_s_corr[i]);
        matrix_free(adam->intermediate_w[i]);
        
        matrix_free(adam->bias_m[i]);
        matrix_free(adam->bias_m_corr[i]);
        matrix_free(adam->bias_s[i]);
        matrix_free(adam->bias_s_corr[i]);
        matrix_free(adam->intermediate_b[i]);

    }

    free(adam->weight_m);
    free(adam->weight_m_corr);
    free(adam->weight_s);
    free(adam->weight_s_corr);
    free(adam->intermediate_w);
    
    free(adam->bias_m);
    free(adam->bias_m_corr);
    free(adam->bias_s);
    free(adam->bias_s_corr);
    free(adam->intermediate_b);
    
    free(optimizer->settings);
    free(optimizer);
}

void update_weights_adam(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    AdamConfig* adam = (AdamConfig*)optimizer->settings;

    Matrix* weight_m = adam->weight_m[layer_idx];
    Matrix* weight_m_corr = adam->weight_m_corr[layer_idx];
    Matrix* weight_s = adam->weight_s[layer_idx];
    Matrix* weight_s_corr = adam->weight_s_corr[layer_idx];
    Matrix* intermediate_w = adam->intermediate_w[layer_idx];

    matrix_scale_inplace(adam->beta_m, weight_m);
    matrix_scale_into(1.0 - adam->beta_m, gradient, intermediate_w);
    matrix_subtract_into(weight_m, intermediate_w, weight_m);

    matrix_scale_inplace(adam->beta_s, weight_s);
    matrix_multiply_into(gradient, gradient, intermediate_w);
    matrix_scale_inplace(1.0 - adam->beta_s, intermediate_w);
    matrix_add_into(weight_s, intermediate_w, weight_s);

    matrix_scale_into(1.0 / (1.0 - pow(adam->beta_m, adam->ctr)), weight_m, weight_m_corr);
    matrix_scale_into(1.0 / (1.0 - pow(adam->beta_s, adam->ctr)), weight_s, weight_s_corr);

    matrix_add_scalar_into(1e-9, weight_s_corr, intermediate_w);
    matrix_apply_inplace(sqrtf, intermediate_w);
    matrix_divide_into(weight_m_corr, intermediate_w, intermediate_w);
    matrix_scale_inplace(adam->learning_rate, intermediate_w);
    matrix_add_into(weights, intermediate_w, weights);
}

void update_bias_adam(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    AdamConfig* adam = (AdamConfig*)optimizer->settings;

    Matrix* bias_m = adam->bias_m[layer_idx];
    Matrix* bias_m_corr = adam->bias_m_corr[layer_idx];
    Matrix* bias_s = adam->bias_s[layer_idx];
    Matrix* bias_s_corr = adam->bias_s_corr[layer_idx];
    Matrix* intermediate_b = adam->intermediate_b[layer_idx];

    matrix_scale_inplace(adam->beta_m, bias_m);
    matrix_scale_into(1.0 - adam->beta_m, gradient, intermediate_b);
    matrix_subtract_into(bias_m, intermediate_b, bias_m);

    matrix_scale_inplace(adam->beta_s, bias_s);
    matrix_multiply_into(gradient, gradient, intermediate_b);
    matrix_scale_inplace(1.0 - adam->beta_s, intermediate_b);
    matrix_add_into(bias_s, intermediate_b, bias_s);

    matrix_scale_into(1.0 / (1.0 - pow(adam->beta_m, adam->ctr)), bias_m, bias_m_corr);
    matrix_scale_into(1.0 / (1.0 - pow(adam->beta_s, adam->ctr)), bias_s, bias_s_corr);

    matrix_add_scalar_into(1e-9, bias_s_corr, intermediate_b);
    matrix_apply_inplace(sqrtf, intermediate_b);
    matrix_divide_into(bias_m_corr, intermediate_b, intermediate_b);
    matrix_scale_inplace(adam->learning_rate, intermediate_b);
    matrix_add_into(bias, intermediate_b, bias);
}