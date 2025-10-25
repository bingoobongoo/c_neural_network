#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cblas.h>
#include <openblas_config.h>

#include "config.h"

typedef struct {
    nn_float* entries;
    int n_rows;
    int n_cols;
} Matrix;

typedef struct {
    uint16_t* entries;
    int n_rows;
    int n_cols;
} Matrix_uint16;

typedef enum {
    VALID,
    FULL
} CorrelationType;

Matrix* matrix_new(int n_rows, int n_cols);
void matrix_free(Matrix* m);
nn_float matrix_get(Matrix* m, int row, int col);
void matrix_assign(Matrix* m, int row, int col, nn_float num);
void matrix_save(Matrix* m, char* file_path);
Matrix* matrix_load(char* file_path);
Matrix* matrix_copy(Matrix* m);
void matrix_copy_into(Matrix* m, Matrix* into);
void matrix_assign_ptr(Matrix** to, Matrix* from);
void matrix_print(Matrix* m);
void matrix_print_dimensions(Matrix* m);
void matrix_zero(Matrix* m);
void matrix_fill(Matrix* m, nn_float num);
void matrix_fill_normal_distribution(Matrix* m, nn_float mean, nn_float std_deviation);
Matrix* matrix_slice_rows(Matrix* m, int start_idx, int slice_size);
void matrix_slice_rows_into(Matrix* m, int start_idx, int slice_size, Matrix* into);
bool matrix_check_dimensions(Matrix* m1, Matrix* m2);
Matrix* matrix_add(Matrix* m1, Matrix* m2);
void matrix_add_into(Matrix* m1, Matrix* m2, Matrix* into);
Matrix* matrix_subtract(Matrix* m1, Matrix* m2);
void matrix_subtract_into(Matrix* m1, Matrix* m2, Matrix* into);
Matrix* matrix_dot(Matrix* m1, Matrix* m2);
void matrix_dot_into(Matrix* m1, Matrix* m2, Matrix* into, bool m1_trans, bool m2_trans);
Matrix* matrix_multiply(Matrix* m1, Matrix* m2);
void matrix_multiply_into(Matrix* m1, Matrix* m2, Matrix* into);
Matrix* matrix_divide(Matrix* m1, Matrix* m2);
void matrix_divide_into(Matrix* m1, Matrix* m2, Matrix* into);
Matrix* matrix_sum_axis(Matrix* m, int axis);
void matrix_sum_axis_into(Matrix* m, int axis, Matrix* into);
nn_float matrix_sum(Matrix* m);
nn_float matrix_average(Matrix* m);
nn_float matrix_min(Matrix* m);
nn_float matrix_max(Matrix* m);
void matrix_argmax_into(Matrix* m, Matrix* into);
Matrix* matrix_multiplicate(Matrix* m, int axis, int n_size);
void matrix_multiplicate_into(Matrix* m, int axis, int n_size, Matrix* into);
Matrix* matrix_apply(nn_float (*func)(nn_float), Matrix* m);
void matrix_apply_into(nn_float (*func)(nn_float), Matrix* m, Matrix* into);
void matrix_apply_inplace(nn_float (*func)(nn_float), Matrix* m);
Matrix* matrix_scale(nn_float scalar, Matrix* m);
void matrix_scale_into(nn_float scalar, Matrix* m, Matrix* into);
void matrix_scale_inplace(nn_float scalar, Matrix* m);
Matrix* matrix_add_scalar(nn_float scalar, Matrix* m);
void matrix_add_scalar_into(nn_float scalar, Matrix* m, Matrix* into);
void matrix_add_scalar_inplace(nn_float scalar, Matrix* m);
Matrix* matrix_transpose(Matrix* m);
void matrix_transpose_into(Matrix* m, Matrix* into);
void matrix_flip_into(Matrix* m, Matrix* into);
void matrix_correlate_into(Matrix* input, Matrix* kernel, Matrix* into, int stride, CorrelationType type);
void matrix_acc_correlate_into(Matrix* input, Matrix* kernel, Matrix* into, int stride, CorrelationType type);
void matrix_convolve_into(Matrix* input, Matrix* kernel, Matrix* into, int stride, CorrelationType type);
void matrix_acc_convolve_valid_into(Matrix* input, Matrix* kflip, Matrix* into, int stride);
void matrix_acc_convolve_full_into(Matrix* input, Matrix* kflip, Matrix* into, Matrix* padding);
void matrix_max_pool_into(Matrix* input, Matrix* into, Matrix_uint16* argmax, int kernel_size, int stride);

Matrix_uint16* matrix_uint16_new(int n_rows, int n_cols);
void matrix_uint16_free(Matrix_uint16* m);
void matrix_uint16_fill(Matrix_uint16* m, uint16_t num);