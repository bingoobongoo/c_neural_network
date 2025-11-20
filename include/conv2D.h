#pragma once

#include "config.h"
#include "matrix.h"
#include "tensor.h"

typedef struct {
    int n_filters;
    int n_filter_channels;
    int filter_size;
    int stride;
    int n_units;
} ConvParams;

typedef struct {
    Tensor4D* output;
    Tensor4D* z;
    Tensor4D* weight;
    Tensor4D* weight_flip;
    Matrix* bias;
    Tensor4D* delta;
    Tensor4D* weight_grad;
    Matrix* bias_grad;

    // auxiliary
    Tensor4D* dL_dA;
    Tensor4D* dA_dZ;
    Tensor4D* padding;

    // im2col auxiliary
    Tensor3D* fp_im2col_input;
    Matrix* fp_im2col_kernel;
    Tensor3D* fp_im2col_output;
    Tensor3D* dL_dW_im2col_input;
    Tensor3D* dL_dW_im2col_kernel;
    Tensor3D* dL_dW_im2col_output;
    Matrix* dL_dW_im2col_output_sum;
    Tensor3D* delta_im2col_input;
    Matrix* delta_im2col_kernel;
    Tensor3D* delta_im2col_output;

    // transpose buffers
    Tensor3D* fp_im2col_output_t;
    Tensor3D* dL_dW_im2col_kernel_t;
    Tensor3D* delta_im2col_output_t;
} ConvCache;