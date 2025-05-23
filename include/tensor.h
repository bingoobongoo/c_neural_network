#pragma once

#include "config.h"
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
void matrix_into_tensor3D(Matrix* m, Tensor3D* t, bool transpose);

Tensor4D* tensor4D_new(int n_rows, int n_cols, int n_channels, int n_filters);
void tensor4D_free(Tensor4D* t);
void tensor4D_copy_into(Tensor4D* t, Tensor4D* into);
void tensor4D_slice_into(Tensor4D* t, int start_idx, int slice_size, Tensor4D* into);
void tensor4D_fill(Tensor4D* t, nn_float num);
void tensor4D_fill_normal_distribution(Tensor4D* t, nn_float mean, nn_float std_deviation);
void tensor4D_print_shape(Tensor4D* t);
Tensor4D* matrix_to_tensor4D(Matrix* m, int n_rows, int n_cols, int n_channels);
void matrix_into_tensor4D(Matrix* m, Tensor4D* t);
void tensor4D_into_matrix_fwise(Tensor4D* t, Matrix* m, bool transpose, bool flipped);
void tensor4D_into_matrix_chwise(Tensor4D* t, Matrix* m, bool transpose, bool flipped);
void kernel_into_im2col_fwise(Tensor4D* kernel, bool flipped, Matrix* kernel_im2col);
void kernel_into_im2col_chwise(Tensor4D* kernel, bool flipped, Matrix* kernel_im2col);
void input_into_im2col_fwise(Tensor4D* input, int filter_idx, Tensor4D* kernel, int stride, CorrelationType corr_type,  Matrix* input_im2col);
void input_into_im2col_chwise(Tensor4D* input, int channel_idx, Tensor4D* kernel, int stride, CorrelationType corr_type,  Matrix* input_im2col);