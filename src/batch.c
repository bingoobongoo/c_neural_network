#include "batch.h"

Batch batchify(Matrix* m, int start_idx, int batch_size, bool is_view) {
    Batch batch;
    batch.is_view = is_view;

    if (is_view) {
        batch.data = matrix_slice_rows_view(m, start_idx, batch_size);
    }
    else {
        batch.data = matrix_slice_rows(m, start_idx, batch_size);
    }

    if (batch.data->n_rows < batch_size) 
        batch.batch_size = batch.data->n_rows;
    else
        batch.batch_size = batch_size;

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