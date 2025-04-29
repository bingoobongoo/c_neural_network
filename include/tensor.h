#pragma once

#include "matrix.h"

typedef struct {
    int n_rows;
    int n_cols;
    int n_channels;
    Matrix** channels;
} Tensor3D;

typedef struct {
    int n_rows;
    int n_cols;
    int n_channels;
    int n_filters;
    Tensor3D** filters;
} Tensor4D;

Tensor3D* tensor3D_new(int n_rows, int n_cols, int n_channels);
void tensor3D_free(Tensor3D* t);
void tensor3D_copy_into(Tensor3D* from, Tensor3D* to);

Tensor4D* tensor4D_new(int n_rows, int n_cols, int n_channels, int n_filters);
void tensor4D_free(Tensor4D* t);
void tensor4D_slice_into(Tensor4D* t, int start_idx, int slice_size, Tensor4D* into);
Tensor4D* matrix_to_tensor4D(Matrix* m, int n_rows, int n_cols, int n_channels);
void conv2d_forward(Tensor4D* in, Tensor4D* filter, Tensor4D* bias, int stride, Tensor4D* out); 
void tensor4D_rot180_into(Tensor4D* t, Tensor4D* into);