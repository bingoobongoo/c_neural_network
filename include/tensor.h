#pragma once

#include "config.h"
#include "matrix.h"

typedef struct {
    int n_rows;
    int n_cols;
    int n_channels;
    Matrix** channels;
    nn_float* entries;
    bool view;
} Tensor3D;

typedef struct {
    int n_rows;
    int n_cols;
    int n_channels;
    int n_filters;
    Tensor3D** filters;
    nn_float* entries;
} Tensor4D;

typedef struct {
    int n_rows;
    int n_cols;
    int n_channels;
    Matrix_uint16** channels;
} Tensor3D_uint16;

typedef struct {
    int n_rows;
    int n_cols;
    int n_channels;
    int n_filters;
    Tensor3D_uint16** filters;
} Tensor4D_uint16;

Tensor3D* tensor3D_new(int n_rows, int n_cols, int n_channels);
Tensor3D* tensor3D_view_new(int n_rows, int n_cols, int n_channels, nn_float* entries);
void tensor3D_free(Tensor3D* t);
void tensor3D_copy_into(Tensor3D* from, Tensor3D* to);
void tensor3D_sum_element_wise_into(Tensor3D* t, Matrix* into);
void tensor3D_acc_correlate_into(Tensor3D* input, Tensor3D* kernel, Matrix* into,  int stride, CorrelationType type);

Tensor4D* tensor4D_new(int n_rows, int n_cols, int n_channels, int n_filters);
void tensor4D_free(Tensor4D* t);
void tensor4D_copy_into(Tensor4D* t, Tensor4D* into);
void tensor4D_slice_into(Tensor4D* t, int start_idx, int slice_size, Tensor4D* into);
void tensor4D_flip_into(Tensor4D* t, Tensor4D* flipped);
void tensor4D_fill(Tensor4D* t, nn_float num);
void tensor4D_zero(Tensor4D* t);
void tensor4D_fill_normal_distribution(Tensor4D* t, nn_float mean, nn_float std_deviation);
void tensor4D_apply_inplace(nn_float (*func)(nn_float), Tensor4D* t);
void tensor4D_scale_inplace(nn_float scalar, Tensor4D* t);
void tensor4D_scale_into(nn_float scalar, Tensor4D* t, Tensor4D* into);
void tensor4D_add_into(Tensor4D* t1, Tensor4D* t2, Tensor4D* into);
void tensor4D_add_scalar_into(nn_float scalar, Tensor4D* t, Tensor4D* into);
void tensor4D_subtract_into(Tensor4D* t1, Tensor4D* t2, Tensor4D* into);
void tensor4D_multiply_into(Tensor4D* t1, Tensor4D* t2, Tensor4D* into);
void tensor4D_divide_into(Tensor4D* t1, Tensor4D* t2, Tensor4D* into);
nn_float tensor4D_sum(Tensor4D* t);
nn_float tensor4D_max(Tensor4D* t);
nn_float tensor4D_min(Tensor4D* t);
nn_float tensor4D_average(Tensor4D* t);
void tensor4D_print_shape(Tensor4D* t);
Tensor4D* matrix_to_tensor4D(Matrix* m, int n_rows, int n_cols, int n_channels);
void matrix_into_tensor4D(Matrix* m, Tensor4D* t);
void tensor4D_into_matrix_fwise(Tensor4D* t, Matrix* m, bool transpose, bool flipped);
void tensor4D_into_matrix_chwise(Tensor4D* t, Matrix* m, bool transpose, bool flipped);
void kernel_into_im2col_fwise(Tensor4D* kernel, bool flipped, Matrix* kernel_im2col);
void kernel_into_im2col_chwise(Tensor4D* kernel, bool flipped, Matrix* kernel_im2col);
void delta_into_im2col_fwise(Tensor4D* delta, int filter_idx, Matrix* im2col);
void input_into_im2col_fwise(Tensor4D* input, int filter_idx, Tensor4D* kernel, int stride, int padding, Matrix* im2col);
size_t tensor3D_get_sizeof_mem_allocated(Tensor3D* t);
size_t tensor4D_get_sizeof_mem_allocated(Tensor4D* t);
size_t tensor4D_uint16_get_sizeof_mem_allocated(Tensor4D_uint16* t);

Tensor3D_uint16* tensor3D_uint16_new(int n_rows, int n_cols, int n_channels);
void tensor3D_uint16_free(Tensor3D_uint16* t);

Tensor4D_uint16* tensor4D_uint16_new(int n_rows, int n_cols, int n_channels, int n_filters);
void tensor4D_uint16_free(Tensor4D_uint16* t);