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
                float sum = matrix_get(into, i, j);
                matrix_assign(into, i, j, sum + matrix_get(t->channels[c], i, j));
            }
        }
    }
}

void tensor3D_correlate_into(Tensor3D* input, Tensor3D* kernel, Tensor3D* into, int stride, CorrelationType type) {
    for (int c=0; c<input->n_channels; c++) {
        matrix_correlate_into(
            input->channels[c],
            kernel->channels[c],
            into->channels[c],
            stride,
            type
        );
    }
}

void matrix_into_tensor3D(Matrix* m, Tensor3D* t, bool transpose) {
    if (transpose) {
        for (int i=0; i<t->n_channels; i++) {
            for (int j=0; j<t->n_rows; j++) {
                for (int k=0; k<t->n_cols; k++) {
                    matrix_assign(t->channels[i], j, k, matrix_get(m, j*t->n_cols+k, i));
                }
            }
        }
    }
    else {
        for (int i=0; i<t->n_channels; i++) {
            for (int j=0; j<t->n_rows; j++) {
                for (int k=0; k<t->n_cols; k++) {
                    matrix_assign(t->channels[i], j, k, matrix_get(m, i, j*t->n_cols+k));
                }
            }
        }
    }
}

void tensor4D_fill(Tensor4D* t, float num) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_fill(t->filters[i]->channels[j], num);
        }
    }
}

void tensor4D_fill_normal_distribution(Tensor4D* t, float mean, float std_deviation) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_fill_normal_distribution(
                t->filters[i]->channels[j],
                mean,
                std_deviation
            );
        }
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
                    matrix_assign(channel_mat, k, l, matrix_get(m, i, idx));
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
                    matrix_assign(channel_mat, k, l, matrix_get(m, i, idx));
                }
            }
        }
    }
}

void tensor4D_into_matrix(Tensor4D* t, Matrix* m, bool transpose) {
    if (transpose) {
        for (int i=0; i<t->n_filters; i++) {
            Tensor3D* t3d = t->filters[i];
    
            for (int j=0; j<t->n_channels; j++) {
                Matrix* channel_mat = t3d->channels[j];
    
                for (int k=0; k<t->n_rows; k++) {
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = j * (t->n_rows * t->n_cols) + k * t->n_cols + l;
                        matrix_assign(m, idx, i, matrix_get(channel_mat, k, l));
                    }
                }
            }
        }
    }
    else {
        for (int i=0; i<t->n_filters; i++) {
            Tensor3D* t3d = t->filters[i];
    
            for (int j=0; j<t->n_channels; j++) {
                Matrix* channel_mat = t3d->channels[j];
    
                for (int k=0; k<t->n_rows; k++) {
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = j * (t->n_rows * t->n_cols) + k * t->n_cols + l;
                        matrix_assign(m, i, idx, matrix_get(channel_mat, k, l));
                    }
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

void kernel_into_im2col(Tensor4D* kernel, Matrix* kernel_im2col) {
    tensor4D_into_matrix(kernel, kernel_im2col, true);
}

void input_into_im2col(Tensor3D* input, Tensor4D* kernel, int stride, Matrix* input_im2col) {
    int out_h = (input->n_rows - kernel->n_rows)/stride + 1;
    int out_w = (input->n_cols - kernel->n_cols)/stride + 1;
    int ker_h = kernel->n_rows;
    int ker_w = kernel->n_cols;

    for (int i=0; i<out_h; i++) {
        for (int j=0; j<out_w; j++) {
            for (int c=0; c<input->n_channels; c++) {
                for (int k=0; k<ker_h; k++) {
                    for (int l=0; l<ker_w; l++) {
                        matrix_assign(
                            input_im2col,
                            i*out_w + j,
                            c*ker_h*ker_w + k*ker_w + l,
                            matrix_get(input->channels[c], i*stride+k, j*stride+l)
                        );
                    }
                }
            }
        }
    }
}

void im2col_correlate(Matrix* input_im2col, Matrix* kernel_im2col, Matrix* im2col_dot, Tensor3D* output) {
    matrix_dot_into(input_im2col, kernel_im2col, im2col_dot);
    matrix_into_tensor3D(im2col_dot, output, true);
}