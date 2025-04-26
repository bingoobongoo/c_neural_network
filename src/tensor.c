#include "tensor.h"

Tensor3D* tensor3D_new(int n_rows, int n_cols, int n_channels) {
    Tensor3D* t = (Tensor3D*)malloc(sizeof(Tensor3D));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->channels = (Matrix**)malloc(n_channels * sizeof(Matrix*));
    for (int i=0; i<n_channels; i++) {
        t->channels[i] = matrix_new(n_rows, n_cols);
    }

    return t;
}

void tensor3D_free(Tensor3D* t) {
    for (int i=0; i<t->n_channels; i++) {
        matrix_free(t->channels[i]);
    }
    free(t->channels);
}

Tensor4D* tensor4D_new(int n_rows, int n_cols, int n_channels, int n_filters) {
    Tensor4D* t = (Tensor4D*)malloc(sizeof(Tensor4D));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->n_filters = n_filters;
    t->filters = (Tensor3D**)malloc(n_filters * sizeof(Tensor3D*));
    for (int i=0; i<n_filters; i++) {
        t->filters[i] = tensor3D_new(n_rows, n_cols, n_channels);
    }

    return t;
}

void tensor4D_free(Tensor4D* t) {
    for (int i=0; i<t->n_filters; i++) {
        tensor3D_free(t->filters[i]);
    }
    free(t->filters);
}