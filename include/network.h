#pragma once

#include <time.h>
#include <sys/time.h>

#include "config.h"
#include "matrix.h"
#include "layer.h"
#include "tensor.h"
#include "activation.h"
#include "loss.h"
#include "batch.h"
#include "optimizer.h"
#include "preprocessing.h"
#include "score.h"
#include "save.h"

struct NeuralNet {
    int n_layers;
    Activation* activation;
    Loss* loss;
    Optimizer* optimizer;
    Score* batch_score;
    Batch* train_batch;
    Batch* label_batch;
    int batch_size;
    Layer** layers;
    bool compiled;
    bool is_cnn;
};

NeuralNet* neural_net_new(Optimizer* opt, ActivationType act_type, nn_float act_param, LossType loss_type, int batch_size);
void neural_net_free(NeuralNet* net);
void neural_net_compile(NeuralNet* net);
void neural_net_link_layers(NeuralNet* net);
void neural_net_info(NeuralNet* net);

void fit(Matrix* x_train, Matrix* y_train, int n_epochs, nn_float validation, NeuralNet* net);
void score(Matrix* x_test, Matrix* y_test, NeuralNet* net);
void confusion_matrix(Matrix* x_test, Matrix* y_test, NeuralNet* net);
void forward_prop(NeuralNet* net, bool training);
void back_prop(NeuralNet* net);
void update_weights(NeuralNet* net);

void add_input_layer(int n_units, NeuralNet* net);
void add_conv_input_layer(int n_rows, int n_cols, int n_channels, NeuralNet* net);
void add_output_layer(int n_units, NeuralNet* net);
void add_dense_layer(int n_units, NeuralNet* net);
void add_conv_layer(int n_filters, int filter_size, int stride, NeuralNet* net);
void add_flatten_layer(NeuralNet* net);
void add_max_pool_layer(int filter_size, int stride, NeuralNet* net);
void add_batch_norm_conv2D_layer(nn_float momentum, NeuralNet* net);
void add_batch_norm_dense_layer(nn_float momentum, NeuralNet* net);

void debug_layers_info(NeuralNet* net);