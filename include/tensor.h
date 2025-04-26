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

Tensor4D* tensor4D_new(int n_rows, int n_cols, int n_channels, int n_filters);
void tensor4D_free(Tensor4D* t);