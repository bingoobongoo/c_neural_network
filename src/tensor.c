#include "tensor.h"

Tensor3D* tensor3D_new(int n_rows, int n_cols, int n_channels) {
    Tensor3D* t = (Tensor3D*)malloc(sizeof(Tensor3D));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->channels = (Matrix**)malloc(n_channels * sizeof(Matrix*));
    for (int i=0; i<n_channels; i++) {
        t->channels[i] = matrix_new(n_rows, n_cols);
        matrix_fill(t->channels[i], 0.0);
    }

    return t;
}

void tensor3D_free(Tensor3D* t) {
    for (int i=0; i<t->n_channels; i++) {
        matrix_free(t->channels[i]);
    }
    free(t->channels);
    free(t);
}

void tensor3D_copy_into(Tensor3D* t, Tensor3D* into) {
    for (int i=0; i<t->n_channels; i++) {
        matrix_copy_into(t->channels[i], into->channels[i]);
    }
}

void tensor3D_sum_element_wise_into(Tensor3D* t, Matrix* into) {
    matrix_fill(into, 0.0);
    for (int c=0; c<t->n_channels; c++) {
        for (int i=0; i<t->n_rows; i++) {
            for (int j=0; j<t->n_cols; j++) {
                into->entries[i][j] += t->channels[c]->entries[i][j];
            }
        }
    }
}

void tensor3D_correlate_into(Tensor3D* input, Tensor3D* kernel, Tensor3D* into, CorrelationType type) {
    for (int c=0; c<input->n_channels; c++) {
        matrix_correlate_into(
            input->channels[c],
            kernel->channels[c],
            into->channels[c],
            type
        );
    }
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
    free(t);
}

void tensor4D_copy_into(Tensor4D* t, Tensor4D* into) {
    for (int i=0; i<t->n_filters; i++) {
        tensor3D_copy_into(t->filters[i], into->filters[i]);
    }
}

void tensor4D_slice_into(Tensor4D* t, int start_idx, int slice_size, Tensor4D* into) {
    if (start_idx >= t->n_filters) {
        printf("Index out of range");
        exit(1);
    }
    if (start_idx + slice_size > t->n_filters) {
        slice_size = t->n_filters - start_idx;
    }
    for (int i=0; i<slice_size; i++) {
        tensor3D_copy_into(t->filters[i+start_idx], into->filters[i]);
    }
}

Tensor4D* matrix_to_tensor4D(Matrix* m, int n_rows, int n_cols, int n_channels) {
    Tensor4D* t = tensor4D_new(n_rows, n_cols, n_channels, m->n_rows);
    for (int i=0; i<t->n_filters; i++) {
        Tensor3D* t3d = t->filters[i];

        for (int j=0; j<n_channels; j++) {
            Matrix* channel_mat = t3d->channels[j];

            for (int k=0; k<t->n_rows; k++) {
                for (int l=0; l<t->n_cols; l++) {
                    int idx = j * (t->n_rows * t->n_cols) + k * t->n_cols + l;
                    channel_mat->entries[k][l] = m->entries[i][idx];
                }
            }
        }
    }

    return t;
}

void matrix_into_tensor4D(Matrix* m, Tensor4D* t) {
    for (int i=0; i<t->n_filters; i++) {
        Tensor3D* t3d = t->filters[i];

        for (int j=0; j<t->n_channels; j++) {
            Matrix* channel_mat = t3d->channels[j];

            for (int k=0; k<t->n_rows; k++) {
                for (int l=0; l<t->n_cols; l++) {
                    int idx = j * (t->n_rows * t->n_cols) + k * t->n_cols + l;
                    channel_mat->entries[k][l] = m->entries[i][idx];
                }
            }
        }
    }
}

void tensor4D_into_matrix(Tensor4D* t, Matrix* m) {
    for (int i=0; i<t->n_filters; i++) {
        Tensor3D* t3d = t->filters[i];

        for (int j=0; j<t->n_channels; j++) {
            Matrix* channel_mat = t3d->channels[j];

            for (int k=0; k<t->n_rows; k++) {
                for (int l=0; l<t->n_cols; l++) {
                    int idx = j * (t->n_rows * t->n_cols) + k * t->n_cols + l;
                    m->entries[i][idx] = channel_mat->entries[k][l];
                }
            }
        }
    }
}

void tensor4D_print_shape(Tensor4D* t) {
    printf("[%d x %d x %d x %d]\n", 
        t->n_filters,
        t->n_channels,
        t->n_rows,
        t->n_cols
    );
}