#pragma once

#include "matrix.h"

typedef struct {
    Matrix* data;
    int batch_size;
    bool is_view;
} Batch;

Batch batchify(Matrix* m, int start_idx, int batch_size, bool is_view);
void batch_free(Batch* batch);