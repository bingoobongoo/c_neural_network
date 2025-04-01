#pragma once

#include "matrix.h"
#include "activation.h"
#include "cost.h"
#include "batch.h"

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
    Matrix* activation;
    Matrix* z;
    Matrix* weight;
    Matrix* bias;
    Matrix* delta;
    Matrix* weight_gradient;
    Matrix* bias_gradient;
    Layer* prev_layer;
    Layer* next_layer;
    NeuralNet* net_backref;
};

struct NeuralNet {
    int n_in_layers;
    int n_ou_layers;
    int n_de_layers;
    Activation* activation;
    Cost* cost;
    int batch_size;
    Layer** layers;
    bool compiled;
};

NeuralNet* neural_net_new(ActivationType activation_type, CostType cost_type, double activation_param, int batch_size);
void neural_net_free(NeuralNet* net);
void neural_net_compile(NeuralNet* net);
void neural_net_link_layers(NeuralNet* net);
void neural_net_info(NeuralNet* net);

void forward_prop(Batch* batch, NeuralNet* net);
void back_prop(Batch* label_batch, NeuralNet* net);

Layer* layer_new(LayerType l_type, int n_units, NeuralNet* net);
void layer_free(Layer* layer);
void add_input_layer(int n_units, NeuralNet* net);
void add_output_layer(int n_units, NeuralNet* net);
void add_deep_layer(int n_units, NeuralNet* net);