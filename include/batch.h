#pragma once

#include "matrix.h"

typedef struct {
    Matrix* data;
    int batch_size;
} Batch;

Batch* batch_new(int batch_size, int n_features);
void batchify_into(Matrix* m, int start_idx, Batch* into);
void batch_free(Batch* batch);