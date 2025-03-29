#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define PI 3.14159265358979323846

typedef struct {
    double** entries;
    int n_rows;
    int n_cols;
} Matrix;

Matrix* matrix_new(int n_row, int n_col);
void matrix_free(Matrix* m);
void matrix_save(Matrix* m, char* file_path);
Matrix* matrix_load(char* file_path);
Matrix* matrix_copy(Matrix* m);
void matrix_print(Matrix* m);
void matrix_print_dimensions(Matrix* m);
void matrix_fill(Matrix* m, double num);
void matrix_fill_normal_distribution(Matrix* m, double mean, double std_deviation);
Matrix* matrix_flatten(Matrix* m, int axis);

bool matrix_check_dimensions(Matrix* m1, Matrix* m2);
Matrix* matrix_add(Matrix* m1, Matrix* m2);
Matrix* matrix_subtract(Matrix* m1, Matrix* m2);
Matrix* matrix_dot(Matrix* m1, Matrix* m2);
Matrix* matrix_multiply(Matrix* m1, Matrix* m2);
Matrix* matrix_apply(double (*func)(double), Matrix* m);
void matrix_apply_inplace(double (*func)(double), Matrix* m);
Matrix* matrix_scale(double scalar, Matrix* m);
void matrix_scale_inplace(double scalar, Matrix* m);
Matrix* matrix_add_scalar(double scalar, Matrix* m);
void matrix_add_scalar_inplace(double scalar, Matrix* m);
Matrix* matrix_transpose(Matrix* m);