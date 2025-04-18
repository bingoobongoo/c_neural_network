#pragma once

#include "matrix.h"
#include "activation.h"
#include "cost.h"
#include "batch.h"
#include "optimizer.h"
#include "preprocessing.h"
#include "score.h"
#include <time.h>
#include <sys/time.h>

typedef struct Layer Layer;
typedef struct NeuralNet NeuralNet;

typedef enum {
    INPUT,
    OUTPUT,
    DEEP,
    UNDEFINED
} LayerType;

struct Layer {
    LayerType l_type;
    int n_units;
    Activation* activation;
    Matrix* output;
    Matrix* z;
    Matrix* weight;
    Matrix* bias;
    Matrix* delta;
    Matrix* weight_gradient;
    Matrix* bias_gradient;

    // auxiliary gradients
    Matrix* dCost_dA; 
    Matrix* dActivation_dZ;
    Matrix* dZ_dW_t;
    Matrix* dZnext_dA_t;
    Matrix* dCost_dZ_col_sum;

    Layer* prev_layer;
    Layer* next_layer;
    NeuralNet* net_backref;
};

struct NeuralNet {
    int n_layers;
    Activation* activation;
    Cost* cost;
    Optimizer* optimizer;
    Score* batch_score;
    Batch* train_batch;
    Batch* label_batch;
    int batch_size;
    Layer** layers;
    bool compiled;
};

NeuralNet* neural_net_new(Optimizer* opt, ActivationType act_type, double act_param, CostType cost_type, int batch_size);
void neural_net_free(NeuralNet* net);
void neural_net_compile(NeuralNet* net);
void neural_net_link_layers(NeuralNet* net);
void neural_net_info(NeuralNet* net);

void fit(Matrix* x_train, Matrix* y_train, int n_epochs, double validation, NeuralNet* net);
void score(Matrix* x_test, Matrix* y_test, NeuralNet* net);
void confusion_matrix(Matrix* x_test, Matrix* y_test, NeuralNet* net);
void forward_prop(NeuralNet* net, bool training);
void back_prop(NeuralNet* net);

Layer* layer_new(LayerType l_type, int n_units, NeuralNet* net);
void layer_free(Layer* layer);
void add_input_layer(int n_units, NeuralNet* net);
void add_output_layer(int n_units, NeuralNet* net);
void add_deep_layer(int n_units, NeuralNet* net);