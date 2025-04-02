#include "batch.h"

Batch batchify(Matrix* m, int start_idx, int batch_size, bool is_view) {
    Batch batch;
    batch.is_view = is_view;

    if (start_idx + batch_size > m->n_rows) 
        batch.batch_size = m->n_rows - start_idx;
    else
        batch.batch_size = batch_size;

    if (is_view) {
        batch.data = matrix_slice_rows_view(m, start_idx, batch.batch_size);
    }
    else {
        batch.data = matrix_slice_rows(m, start_idx, batch.batch_size);
    }

    return batch;
}

void batch_free(Batch* batch) {
    if (batch->is_view) {
        matrix_free_view(batch->data);
    }
    else {
        matrix_free(batch->data);
    }
}