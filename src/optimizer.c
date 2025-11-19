#include "optimizer.h"

Optimizer* optimizer_sgd_new(nn_float learning_rate) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_dense_weights = update_dense_weights_sgd;
    opt->update_conv_weights = update_conv_weights_sgd;
    opt->update_bias = update_bias_sgd;
    opt->update_batch_norm_gamma = update_dense_weights_sgd;
    opt->update_batch_norm_beta = update_bias_sgd;
    opt->optimizer_free = optimizer_sgd_free;
    opt->optimizer_print_info = optimizer_sgd_print_info;
    opt->optimizer_get_mem_allocated = optimizer_sgd_get_mem_allocated;
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

void update_dense_weights_sgd(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    SGDConfig* sgd = (SGDConfig*)optimizer->settings;
    matrix_scale_inplace(sgd->learning_rate, gradient);
    matrix_subtract_into(weights, gradient, weights);
}

void update_conv_weights_sgd(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx) {
    SGDConfig* sgd = (SGDConfig*)optimizer->settings;
    tensor4D_scale_inplace(sgd->learning_rate, gradient);
    tensor4D_subtract_into(weights, gradient, weights);
}

void update_bias_sgd(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    SGDConfig* sgd = (SGDConfig*)optimizer->settings;
    matrix_scale_inplace(sgd->learning_rate, gradient);
    matrix_subtract_into(bias, gradient, bias);
}

void optimizer_sgd_print_info(Optimizer* optimizer) {
    SGDConfig* sgd = (SGDConfig*)optimizer->settings;
    printf("Optimizer: %s\n", optimizer->name);
    printf("Learning rate: %f\n", sgd->learning_rate);
    printf(
        "Optimizer memory allocated: %ld B\n", 
        optimizer->optimizer_get_mem_allocated(optimizer)
    );
    printf("------------------------------------\n");
}

size_t optimizer_sgd_get_mem_allocated(Optimizer* optimizer) {
    size_t size = 0;
    if (optimizer == NULL) return size;

    size += sizeof(*optimizer);

    SGDConfig* sgd = (SGDConfig*)optimizer->settings;
    size += sizeof(*sgd);

    return size;
}

Optimizer* optimizer_momentum_new(nn_float learning_rate, nn_float beta, bool nesterov) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_dense_weights = update_dense_weights_momentum;
    opt->update_conv_weights = update_conv_weights_momentum;
    opt->update_bias = update_bias_momentum;
    opt->update_batch_norm_gamma = update_dense_weights_momentum;
    opt->update_batch_norm_beta = update_bias_momentum;
    opt->optimizer_free = optimizer_momentum_free;
    opt->optimizer_print_info = optimizer_momentum_print_info;
    opt->optimizer_get_mem_allocated = optimizer_momentum_get_mem_allocated;

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
        tensor4D_free(mom->weight_momentum[i]);
        matrix_free(mom->bias_momentum[i]);
    }

    free(mom->weight_momentum);
    free(mom->bias_momentum);

    free(optimizer->settings);
    free(optimizer);
}

void update_dense_weights_momentum(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;

    Matrix* weight_momentum = mom->weight_momentum[layer_idx]->filters[0]->channels[0];

    nn_float* w = weights->entries;
    nn_float* g = gradient->entries;
    nn_float* w_m = weight_momentum->entries;
    nn_float lr = mom->learning_rate;
    nn_float beta = mom->beta;
    int n = weights->n_rows * weights->n_cols;

    if (optimizer->type == NESTEROV) {
        for (int i=0; i<n; i++) {
            w[i] -= w_m[i];
            w_m[i] = w_m[i] * beta - g[i] * lr;
            w[i] += w_m[i];
        }
    }
    else {
        for (int i=0; i<n; i++) {
            w_m[i] = w_m[i] * beta - g[i] * lr;
            w[i] += w_m[i];
        }
    } 
}

void update_conv_weights_momentum(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;

    Tensor4D* weight_momentum = mom->weight_momentum[layer_idx];

    nn_float lr = mom->learning_rate;
    nn_float beta = mom->beta;
    int n = weights->n_rows * weights->n_cols;

    if (optimizer->type == NESTEROV) {
        for (int f=0; f<weights->n_filters; f++) {
            for (int c=0; c<weights->n_channels; c++) {
                nn_float* w = weights->filters[f]->channels[c]->entries;
                nn_float* g = gradient->filters[f]->channels[c]->entries;
                nn_float* w_m = weight_momentum->filters[f]->channels[c]->entries;
                for (int i=0; i<n; i++) {
                    w[i] -= w_m[i];
                    w_m[i] = w_m[i] * beta - g[i] * lr;
                    w[i] += w_m[i];
                }
            }
        }
    }
    else {
        for (int f=0; f<weights->n_filters; f++) {
            for (int c=0; c<weights->n_channels; c++) {
                nn_float* w = weights->filters[f]->channels[c]->entries;
                nn_float* g = gradient->filters[f]->channels[c]->entries;
                nn_float* w_m = weight_momentum->filters[f]->channels[c]->entries;
                for (int i=0; i<n; i++) {
                    w_m[i] = w_m[i] * beta - g[i] * lr;
                    w[i] += w_m[i];
                }
            }
        }
    }
}

void update_bias_momentum(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;

    Matrix* bias_momentum = mom->bias_momentum[layer_idx];

    nn_float* b = bias->entries;
    nn_float* g = gradient->entries;
    nn_float* b_m = bias_momentum->entries;
    nn_float lr = mom->learning_rate;
    nn_float beta = mom->beta;
    int n = bias->n_rows * bias->n_cols;

    if (optimizer->type == NESTEROV) {
        for (int i=0; i<n; i++) {
            b[i] -= b_m[i];
            b_m[i] = b_m[i] * beta - g[i] * lr;
            b[i] += b_m[i];
        }
    }
    else {
        for (int i=0; i<n; i++) {
            b_m[i] = b_m[i] * beta - g[i] * lr;
            b[i] += b_m[i];
        }
    } 
}

void pre_update_dense_weights_nesterov(Matrix* weight, Optimizer* optimizer, int layer_idx) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;
    matrix_add_into(
        weight,
        mom->weight_momentum[layer_idx]->filters[0]->channels[0],
        weight
    );
}

void pre_update_conv_weights_nesterov(Tensor4D* weight, Optimizer* optimizer, int layer_idx) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;
    tensor4D_add_into(
        weight,
        mom->weight_momentum[layer_idx],
        weight
    );
}

void pre_update_bias_nesterov(Matrix* bias, Optimizer* optimizer, int layer_idx) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;
    matrix_add_into(
        bias,
        mom->bias_momentum[layer_idx],
        bias
    );
}

void optimizer_momentum_print_info(Optimizer* optimizer) {
    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;
    printf("Optimizer: %s\n", optimizer->name);
    printf("Learning rate: %f\n", mom->learning_rate);
    printf("Beta (momentum coefficient): %f\n", mom->beta);
    printf(
        "Optimizer memory allocated: %ld B\n",
        optimizer->optimizer_get_mem_allocated(optimizer)
    );
    printf("------------------------------------\n");
}

size_t optimizer_momentum_get_mem_allocated(Optimizer* optimizer) {
    size_t size = 0;
    if (optimizer == NULL) return size;

    size += sizeof(*optimizer);

    MomentumConfig* mom = (MomentumConfig*)optimizer->settings;
    size += sizeof(*mom);
    size += mom->n_layers * sizeof(*(mom->weight_momentum));
    size += mom->n_layers * sizeof(*(mom->bias_momentum));

    for (int i=0; i<mom->n_layers; i++) {
        if (mom->weight_momentum[i] != NULL) {
            size += tensor4D_get_sizeof_mem_allocated(mom->weight_momentum[i]);
        }
        if (mom->bias_momentum[i] != NULL) {
            size += matrix_get_sizeof_mem_allocated(mom->bias_momentum[i]);
        }
    }

    return size;
}

Optimizer* optimizer_adagrad_new(nn_float learning_rate) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_dense_weights = update_dense_weights_adagrad;
    opt->update_conv_weights = update_conv_weights_adagrad;
    opt->update_bias = update_bias_adagrad;
    opt->update_batch_norm_gamma = update_dense_weights_adagrad;
    opt->update_batch_norm_beta = update_bias_adagrad;
    opt->optimizer_free = optimizer_adagrad_free;
    opt->optimizer_print_info = optimizer_adagrad_print_info;
    opt->optimizer_get_mem_allocated = optimizer_adagrad_get_mem_allocated;
    opt->type = ADAGRAD;
    opt->name = "AdaGrad";

    AdaGradConfig* ada = (AdaGradConfig*)malloc(sizeof(AdaGradConfig));
    ada->learning_rate = learning_rate;
    ada->n_layers = -1;
    ada->weight_s = NULL;
    ada->bias_s = NULL;

    opt->settings = ada;

    return opt;
}

void optimizer_adagrad_free(Optimizer* optimizer) {
    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;

    for (int i=0; i<ada->n_layers; i++) {
        tensor4D_free(ada->weight_s[i]);
        matrix_free(ada->bias_s[i]);
    }

    free(ada->weight_s);
    free(ada->bias_s);

    free(optimizer->settings);
    free(optimizer);
}

void update_dense_weights_adagrad(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;

    Matrix* weight_s = ada->weight_s[layer_idx]->filters[0]->channels[0];
    
    nn_float* w = weights->entries;
    nn_float* g = gradient->entries;
    nn_float* s = weight_s->entries;

    int n = weights->n_rows * weights->n_cols;
    nn_float lr = ada->learning_rate;

    for (int i=0; i<n; i++) {
        s[i] += g[i] * g[i];

        #ifdef SINGLE_PRECISION
        nn_float denom = sqrtf(s[i] + 1e-5);
        #elif defined(DOUBLE_PRECISION)
        nn_float denom = sqrt(s[i] + 1e-5);
        #endif

        nn_float step = lr * g[i] / denom;
        w[i] -= step;
    }
}

void update_conv_weights_adagrad(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx) {
    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;

    Tensor4D* weight_s = ada->weight_s[layer_idx];
    int n = weights->n_rows * weights->n_cols;
    nn_float lr = ada->learning_rate;

    for (int f=0; f<weights->n_filters; f++) {
        for (int c=0; c<weights->n_channels; c++) {
            nn_float* w = weights->filters[f]->channels[c]->entries;
            nn_float* g = gradient->filters[f]->channels[c]->entries;
            nn_float* s = weight_s->filters[f]->channels[c]->entries;

            for (int i=0; i<n; i++) {
                s[i] += g[i] * g[i];

                #ifdef SINGLE_PRECISION
                nn_float denom = sqrtf(s[i] + 1e-5);
                #elif defined(DOUBLE_PRECISION)
                nn_float denom = sqrt(s[i] + 1e-5);
                #endif

                nn_float step = lr * g[i] / denom;
                w[i] -= step;
            }
        }
    }

    
}

void update_bias_adagrad(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;

    Matrix* bias_s = ada->bias_s[layer_idx];
    nn_float* b = bias->entries;
    nn_float* g = gradient->entries;
    nn_float* s = bias_s->entries;

    int n = bias->n_rows * bias->n_cols;
    nn_float lr = ada->learning_rate;

    for (int i=0; i<n; i++) {
        s[i] += g[i] * g[i];

        #ifdef SINGLE_PRECISION
        nn_float denom = sqrtf(s[i] + 1e-5);
        #elif defined(DOUBLE_PRECISION)
        nn_float denom = sqrt(s[i] + 1e-5);
        #endif

        nn_float step = lr * g[i] / denom;
        b[i] -= step;
    }

}

void optimizer_adagrad_print_info(Optimizer* optimizer) {
    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;
    printf("Optimizer: %s\n", optimizer->name);
    printf("Learning rate: %f\n", ada->learning_rate);
    printf(
        "Optimizer memory allocated: %ld B\n",
        optimizer->optimizer_get_mem_allocated(optimizer)
    );
    printf("------------------------------------\n");
}

size_t optimizer_adagrad_get_mem_allocated(Optimizer* optimizer) {
    size_t size = 0;
    if (optimizer == NULL) return size;

    size += sizeof(*optimizer);

    AdaGradConfig* ada = (AdaGradConfig*)optimizer->settings;
    size += sizeof(*ada);
    size += ada->n_layers * sizeof(*(ada->weight_s));
    size += ada->n_layers * sizeof(*(ada->bias_s));

    for (int i=0; i<ada->n_layers; i++) {
        if (ada->weight_s[i] != NULL) {
            size += tensor4D_get_sizeof_mem_allocated(ada->weight_s[i]);
        }
        if (ada->bias_s[i] != NULL) {
            size += matrix_get_sizeof_mem_allocated(ada->bias_s[i]);
        }
    }

    return size;
}

Optimizer* optimizer_adam_new(nn_float learning_rate, nn_float beta_m, nn_float beta_s) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->update_dense_weights = update_dense_weights_adam;
    opt->update_conv_weights = update_conv_weights_adam;
    opt->update_bias = update_bias_adam;
    opt->update_batch_norm_gamma = update_dense_weights_adam;
    opt->update_batch_norm_beta = update_bias_adam;
    opt->optimizer_free = optimizer_adam_free;
    opt->optimizer_print_info = optimizer_adam_print_info;
    opt->optimizer_get_mem_allocated = optimizer_adam_get_mem_allocated;
    opt->type = ADAM;
    opt->name = "Adam";

    AdamConfig* adam = (AdamConfig*)malloc(sizeof(AdamConfig));
    adam->learning_rate = learning_rate;
    adam->beta_m = beta_m;
    adam->beta_s = beta_s;
    adam->n_layers = -1;
    adam->beta_m_pow = (nn_float)1.0;
    adam->beta_s_pow = (nn_float)1.0;

    adam->weight_m = NULL;
    adam->bias_m = NULL;

    adam->weight_s = NULL;
    adam->bias_s = NULL;

    opt->settings = adam;

    return opt;
}

void optimizer_adam_free(Optimizer* optimizer) {
    AdamConfig* adam = (AdamConfig*)optimizer->settings;

    for (int i=1; i<adam->n_layers; i++) {
        tensor4D_free(adam->weight_m[i]);
        tensor4D_free(adam->weight_s[i]);
        
        matrix_free(adam->bias_m[i]);
        matrix_free(adam->bias_s[i]);
    }

    free(adam->weight_m);
    free(adam->weight_s);
    
    free(adam->bias_m);
    free(adam->bias_s);
    
    free(optimizer->settings);
    free(optimizer);
}

void update_dense_weights_adam(Matrix* weights, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    AdamConfig* adam = (AdamConfig*)optimizer->settings;

    Matrix* weight_m = adam->weight_m[layer_idx]->filters[0]->channels[0];
    Matrix* weight_s = adam->weight_s[layer_idx]->filters[0]->channels[0];

    nn_float* w = weights->entries;
    nn_float* g = gradient->entries;
    nn_float* m = weight_m->entries;
    nn_float* s = weight_s->entries;

    int n = weights->n_rows * weights->n_cols;

    nn_float beta_m = adam->beta_m;
    nn_float beta_s = adam->beta_s;
    nn_float lr = adam->learning_rate;

    nn_float beta_m_pow = adam->beta_m_pow;
    nn_float beta_s_pow = adam->beta_s_pow;
    nn_float inv_bias_m = (nn_float)1.0 / ((nn_float)1.0 - beta_m_pow);
    nn_float inv_bias_s = (nn_float)1.0 / ((nn_float)1.0 - beta_s_pow);

    for (int i=0; i<n; i++) {
        m[i] = beta_m * m[i] + ((nn_float)1.0 - beta_m) * g[i];
        s[i] = beta_s * s[i] + ((nn_float)1.0 - beta_s) * g[i] * g[i];

        nn_float m_hat = m[i] * inv_bias_m;
        nn_float s_hat = s[i] * inv_bias_s;

        #ifdef SINGLE_PRECISION
        nn_float denom = sqrtf(s_hat + 1e-5);
        #elif defined(DOUBLE_PRECISION)
        nn_float denom = sqrt(s_hat + 1e-5);
        #endif

        nn_float step = lr * (m_hat / denom);

        w[i] -= step;
    }
}

void update_conv_weights_adam(Tensor4D* weights, Tensor4D* gradient, Optimizer* optimizer, int layer_idx) {
    AdamConfig* adam = (AdamConfig*)optimizer->settings;

    Tensor4D* weight_m = adam->weight_m[layer_idx];
    Tensor4D* weight_s = adam->weight_s[layer_idx];

    nn_float beta_m = adam->beta_m;
    nn_float beta_s = adam->beta_s;
    nn_float lr = adam->learning_rate;

    nn_float beta_m_pow = adam->beta_m_pow;
    nn_float beta_s_pow = adam->beta_s_pow;
    nn_float inv_bias_m = (nn_float)1.0 / ((nn_float)1.0 - beta_m_pow);
    nn_float inv_bias_s = (nn_float)1.0 / ((nn_float)1.0 - beta_s_pow);

    for (int f=0; f<weights->n_filters; f++) {
        for (int c=0; c<weights->n_channels; c++) {
            nn_float* w = weights->filters[f]->channels[c]->entries;
            nn_float* g = gradient->filters[f]->channels[c]->entries;
            nn_float* m = weight_m->filters[f]->channels[c]->entries;
            nn_float* s = weight_s->filters[f]->channels[c]->entries;

            int n = weights->n_rows * weights->n_cols;

            for (int i=0; i<n; i++) {
                m[i] = beta_m * m[i] + ((nn_float)1.0 - beta_m) * g[i];
                s[i] = beta_s * s[i] + ((nn_float)1.0 - beta_s) * g[i] * g[i];

                nn_float m_hat = m[i] * inv_bias_m;
                nn_float s_hat = s[i] * inv_bias_s;

                #ifdef SINGLE_PRECISION
                nn_float denom = sqrtf(s_hat + 1e-5);
                #elif defined(DOUBLE_PRECISION)
                nn_float denom = sqrt(s_hat + 1e-5);
                #endif

                nn_float step = lr * (m_hat / denom);

                w[i] -= step;
            }
        }
    }

}

void update_bias_adam(Matrix* bias, Matrix* gradient, Optimizer* optimizer, int layer_idx) {
    AdamConfig* adam = (AdamConfig*)optimizer->settings;

    Matrix* bias_m = adam->bias_m[layer_idx];
    Matrix* bias_s = adam->bias_s[layer_idx];

    nn_float* b = bias->entries;
    nn_float* g = gradient->entries;
    nn_float* m = bias_m->entries;
    nn_float* s = bias_s->entries;

    int n = bias->n_rows * bias->n_cols;

    nn_float beta_m = adam->beta_m;
    nn_float beta_s = adam->beta_s;
    nn_float lr = adam->learning_rate;

    nn_float beta_m_pow = adam->beta_m_pow;
    nn_float beta_s_pow = adam->beta_s_pow;
    nn_float inv_bias_m = (nn_float)1.0 / ((nn_float)1.0 - beta_m_pow);
    nn_float inv_bias_s = (nn_float)1.0 / ((nn_float)1.0 - beta_s_pow);

    for (int i=0; i<n; i++) {
        m[i] = beta_m * m[i] + ((nn_float)1.0 - beta_m) * g[i];
        s[i] = beta_s * s[i] + ((nn_float)1.0 - beta_s) * g[i] * g[i];

        nn_float m_hat = m[i] * inv_bias_m;
        nn_float s_hat = s[i] * inv_bias_s;

        #ifdef SINGLE_PRECISION
        nn_float denom = sqrtf(s_hat + 1e-5);
        #elif defined(DOUBLE_PRECISION)
        nn_float denom = sqrt(s_hat + 1e-5);
        #endif

        nn_float step = lr * (m_hat / denom);

        b[i] -= step;
    }
}

void optimizer_adam_print_info(Optimizer* optimizer) {
    AdamConfig* adam = (AdamConfig*)optimizer->settings;
    printf("Optimizer: %s\n", optimizer->name);
    printf("Learning rate: %f\n", adam->learning_rate);
    printf("Beta_m (first momentum decay): %f\n", adam->beta_m);
    printf("Beta_s (second momentum decay): %f\n", adam->beta_s);
    printf(
        "Optimizer memory allocated: %ld B\n",
        optimizer->optimizer_get_mem_allocated(optimizer)
    );
    printf("------------------------------------\n");
}

size_t optimizer_adam_get_mem_allocated(Optimizer* optimizer) {
    size_t size = 0;
    if (optimizer == NULL) return size;

    size += sizeof(*optimizer);

    AdamConfig* adam = (AdamConfig*)optimizer->settings;
    size += sizeof(*adam);
    size += adam->n_layers * sizeof(*(adam->weight_m));
    size += adam->n_layers * sizeof(*(adam->weight_s));

    size += adam->n_layers * sizeof(*(adam->bias_m));
    size += adam->n_layers * sizeof(*(adam->bias_s));

    for (int i=0; i<adam->n_layers; i++) {
        if (adam->weight_m[i] != NULL) {
            size += tensor4D_get_sizeof_mem_allocated(adam->weight_m[i]);
        }
        if (adam->weight_s[i] != NULL) {
            size += tensor4D_get_sizeof_mem_allocated(adam->weight_s[i]);
        }
        if (adam->bias_m[i] != NULL) {
            size += matrix_get_sizeof_mem_allocated(adam->bias_m[i]);
        }
        if (adam->bias_s[i] != NULL) {
            size += matrix_get_sizeof_mem_allocated(adam->bias_s[i]);
        }
    }

    return size;
}