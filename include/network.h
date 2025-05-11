#pragma once

#include "matrix.h"
#include "tensor.h"
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
    CONV_2D_INPUT,
    CONV_2D,
    MAX_POOL,
    FLATTEN,
    UNDEFINED
} LayerType;


typedef struct {
    int n_units;
} DenseParams;

typedef struct {
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
} DenseCache;

typedef struct {
    int n_filters;
    int filter_size;
    int stride;
    int n_units;
} ConvParams;

typedef struct {
    Tensor4D* output;
    Tensor4D* z;
    Tensor4D* filter;
    Tensor4D* bias;
    Tensor4D* delta;
    Tensor4D* filter_gradient;
    Tensor4D* bias_gradient;

    // auxiliary
    Tensor4D* dCost_dA;
    Tensor4D* dActivation_dZ;
    Matrix* input_im2col;
    Matrix* kernel_im2col;
    Matrix* im2col_dot;
} ConvCache;

typedef struct {
    int n_units;
} FlattenParams;

typedef struct {
    Matrix* output;
    Matrix* dCost_dA_matrix;
    Matrix* dZnext_dA_t;
} FlattenCache;

typedef union {
    DenseParams dense;
    ConvParams conv;
    FlattenParams flat;
} LayerParams;

typedef union {
    DenseCache dense;
    ConvCache conv;
    FlattenCache flat;
} LayerCache;

struct Layer {
    LayerType l_type;
    LayerParams params;
    LayerCache cache;
    Activation* activation;
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
    bool is_cnn;
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

Layer* layer_new(LayerType l_type, NeuralNet* net);
void layer_free(Layer* layer);

int layer_get_n_units(Layer* layer);
Matrix* layer_get_output_matrix(Layer* layer);
Tensor4D* layer_get_output_tensor4D(Layer* layer);
Matrix* layer_get_delta_matrix(Layer* layer);
Tensor4D* layer_get_delta_tensor4D(Layer* layer);

void add_input_layer(int n_units, NeuralNet* net);
void add_conv_input_layer(int n_rows, int n_cols, int n_channels, NeuralNet* net);
void add_output_layer(int n_units, NeuralNet* net);
void add_deep_layer(int n_units, NeuralNet* net);
void add_conv_layer(int n_filters, int filter_size, int stride, NeuralNet* net);
void add_flatten_layer(NeuralNet* net);
void add_max_pool_layer(int filter_size, int stride, NeuralNet* net);

void layer_deep_compile(Layer* l, NeuralNet* net);
void layer_output_compile(Layer* l, NeuralNet* net);
void layer_conv2D_compile(Layer* l, NeuralNet* net);
void layer_flatten_compile(Layer* l, NeuralNet* net);
void layer_max_pool_compile(Layer* l, NeuralNet* net);