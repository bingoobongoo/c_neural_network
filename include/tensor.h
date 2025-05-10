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
void tensor3D_sum_element_wise_into(Tensor3D* t, Matrix* into);
void tensor3D_correlate_into(Tensor3D* input, Tensor3D* kernel, Tensor3D* into,  int stride, CorrelationType type);

Tensor4D* tensor4D_new(int n_rows, int n_cols, int n_channels, int n_filters);
void tensor4D_free(Tensor4D* t);
void tensor4D_copy_into(Tensor4D* t, Tensor4D* into);
void tensor4D_slice_into(Tensor4D* t, int start_idx, int slice_size, Tensor4D* into);
void tensor4D_fill(Tensor4D* t, double num);
void tensor4D_fill_normal_distribution(Tensor4D* t, double mean, double std_deviation);
Tensor4D* matrix_to_tensor4D(Matrix* m, int n_rows, int n_cols, int n_channels);
void matrix_into_tensor4D(Matrix* m, Tensor4D* t);
void tensor4D_into_matrix(Tensor4D* t, Matrix* m);
void tensor4D_print_shape(Tensor4D* t);