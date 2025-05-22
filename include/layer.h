#pragma once

#include "config.h"
#include "dense.h"
#include "conv2D.h"
#include "flatten.h"
#include "activation.h"
#include "optimizer.h"
#include "cost.h"

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

struct Layer{
    LayerType l_type;
    LayerParams params;
    LayerCache cache;
    Activation* activation;
    Layer* prev_layer;
    Layer* next_layer;
    NeuralNet* net_backref;
    int layer_idx;
};

Layer* layer_new(LayerType l_type, NeuralNet* net);
void layer_free(Layer* layer);
int layer_get_n_units(Layer* layer);
Matrix* layer_get_output_matrix(Layer* layer);
Tensor4D* layer_get_output_tensor4D(Layer* layer);
Matrix* layer_get_delta_matrix(Layer* layer);
Tensor4D* layer_get_delta_tensor4D(Layer* layer);

void layer_deep_compile(Layer* l, ActivationType act_type, int act_param, int batch_size);
void layer_output_compile(Layer* l, Cost* cost, int batch_size);
void layer_conv2D_compile(Layer* l, ActivationType act_type, int act_param, int batch_size);
void layer_flatten_compile(Layer* l, int batch_size);
void layer_max_pool_compile(Layer* l, int batch_size);

void layer_input_fp(Layer* l, Batch* train_batch, int batch_size);
void layer_conv2D_input_fp(Layer* l, Batch* train_batch, int batch_size);
void layer_deep_fp(Layer* l, int batch_size);
void layer_output_fp(Layer* l, Batch* label_batch, int batch_size);
void layer_conv2D_fp(Layer* l, int batch_size);
void layer_flatten_fp(Layer* l, int batch_size);
void layer_max_pool_fp(Layer* l, int batch_size);

void layer_output_bp(Layer* l, Cost* cost, Batch* label_batch, int batch_size);
void layer_deep_bp(Layer* l, int batch_size);
void layer_conv2D_bp(Layer* l, int batch_size);
void layer_flatten_bp(Layer* l, int batch_size);
void layer_max_pool_bp(Layer* l, int batch_size);

void layer_deep_update_weights(Layer* l, Optimizer* opt);
void layer_conv2D_update_weights(Layer* l, Optimizer* opt);