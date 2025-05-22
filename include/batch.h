#pragma once

#include "config.h"
#include "matrix.h"
#include "tensor.h"

typedef enum {
    MATRIX,
    TENSOR
} BatchType;

typedef union {
    Matrix* matrix;
    Tensor4D* tensor;
} BatchData;

typedef struct {
    BatchType type;
    int batch_size;
    BatchData data;
} Batch;

Batch* batch_matrix_new(int batch_size, int n_features);
Batch* batch_tensor_new(int batch_size, int n_rows, int n_cols, int n_channels);

void batchify_matrix_into(Matrix* m, int start_idx, Batch* into);
void batchify_tensor_into(Tensor4D* t, int strat_idx, Batch* into);

void batch_free(Batch* batch);