#include "batch.h"

Batch* batch_new(int batch_size, int n_features) {
    Batch* b = (Batch*)malloc(sizeof(Batch));
    b->batch_size = batch_size;
    b->data = matrix_new(batch_size, n_features);

    return b;
}

void batchify_into(Matrix* m, int start_idx, Batch* into) {
    if (start_idx + into->batch_size > m->n_rows) 
        into->batch_size = m->n_rows - start_idx;

    matrix_slice_rows_into(m, start_idx, into->batch_size, into->data);
}

void batch_free(Batch* batch) {
    matrix_free(batch->data);
    free(batch);
}