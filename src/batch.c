#include "batch.h"

Batch* batch_matrix_new(int batch_size, int n_features) {
    Batch* b = (Batch*)malloc(sizeof(Batch));
    b->type = MATRIX;
    b->batch_size = batch_size;
    b->data.matrix = matrix_new(batch_size, n_features);

    return b;
}

Batch* batch_tensor_new(int batch_size, int n_rows, int n_cols, int n_channels) {
    Batch* b = (Batch*)malloc(sizeof(Batch));
    b->type = TENSOR;
    b->batch_size = batch_size;
    b->data.tensor = tensor4D_new(n_rows, n_cols, n_channels, batch_size);

    return b;
}

void batchify_matrix_into(Matrix* m, int start_idx, Batch* into) {
    if (start_idx + into->batch_size > m->n_rows) 
        into->batch_size = m->n_rows - start_idx;

    matrix_slice_rows_into(m, start_idx, into->batch_size, into->data.matrix);
}

void batchify_tensor_into(Tensor4D* t, int start_idx, Batch* into) {
    if (start_idx + into->batch_size > t->n_filters)
        into->batch_size = t->n_filters - start_idx;
    
    tensor4D_slice_into(t, start_idx, into->batch_size, into->data.tensor);
}

void batch_free(Batch* batch) {
    switch (batch->type)
    {
    case MATRIX:
        matrix_free(batch->data.matrix);
        break;
    
    case TENSOR:
        tensor4D_free(batch->data.tensor);
        break;
    }

    free(batch);
}