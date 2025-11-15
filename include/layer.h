#pragma once

#include "config.h"
#include "dense.h"
#include "conv2D.h"
#include "flatten.h"
#include "max_pool.h"
#include "activation.h"
#include "optimizer.h"
#include "cost.h"
#include "bias.h"

typedef struct Layer Layer;
typedef struct NeuralNet NeuralNet;

typedef enum {
    INPUT,
    OUTPUT,
    DENSE,
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
    MaxPoolParams max_pool;
} LayerParams;

typedef union {
    DenseCache dense;
    ConvCache conv;
    FlattenCache flat;
    MaxPoolCache max_pool;
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
void layer_free(Layer* l);
int layer_get_n_units(Layer* l);
Matrix* layer_get_output_matrix(Layer* l);
Tensor4D* layer_get_output_tensor4D(Layer* l);
Matrix* layer_get_delta_matrix(Layer* l);
Tensor4D* layer_get_delta_tensor4D(Layer* l);
unsigned int layer_get_sizeof_mem_allocated(Layer* l);

void layer_dense_compile(Layer* l, ActivationType act_type, int act_param, int batch_size);
void layer_output_compile(Layer* l, Cost* cost, int batch_size);
void layer_conv2D_compile(Layer* l, ActivationType act_type, int act_param, int batch_size);
void layer_flatten_compile(Layer* l, int batch_size);
void layer_max_pool_compile(Layer* l, int batch_size);

void layer_input_fp(Layer* l, Batch* train_batch, int batch_size);
void layer_conv2D_input_fp(Layer* l, Batch* train_batch, int batch_size);
void layer_dense_fp(Layer* l, int batch_size);
void layer_output_fp(Layer* l, Batch* label_batch, int batch_size);
void layer_conv2D_fp(Layer* l, int batch_size);
void layer_flatten_fp(Layer* l, int batch_size);
void layer_max_pool_fp(Layer* l, int batch_size);

void layer_output_bp(Layer* l, Cost* cost, Batch* label_batch, int batch_size);
void layer_dense_bp(Layer* l, int batch_size);
void layer_conv2D_bp(Layer* l, int batch_size);
void layer_conv2D_bp_delta_from_flatten(Layer* l, int batch_size);
void layer_conv2d_bp_delta_from_max_pool(Layer* l, int batch_size);
void layer_conv2d_bp_delta_from_conv2d(Layer* l, int batch_size);
void layer_flatten_bp(Layer* l, int batch_size);
void layer_max_pool_bp(Layer* l, int batch_size);

void layer_dense_update_weights(Layer* l, Optimizer* opt);
void layer_conv2D_update_weights(Layer* l, Optimizer* opt);

unsigned long layer_output_get_sizeof_mem_allocated(Layer* l);
unsigned long layer_dense_get_sizeof_mem_allocated(Layer* l);
unsigned long layer_conv2D_get_sizeof_mem_allocated(Layer* l);
unsigned long layer_flatten_get_sizeof_mem_allocated(Layer* l);
unsigned long layer_max_pool_get_sizeof_mem_allocated(Layer* l);