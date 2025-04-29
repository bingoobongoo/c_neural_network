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

void conv2d_forward(Tensor4D* in, Tensor4D* filter, Tensor4D* bias, int stride, Tensor4D* out) {
    int batch_size = in->n_filters;
    int input_h = in->n_rows;
    int input_w = in->n_cols;
    int input_c = in->n_channels;

    int n_filters = filter->n_filters;
    int filter_h = filter->n_rows;
    int filter_w = filter->n_cols;

    int output_h = out->n_rows;
    int output_w = out->n_cols;

    for (int n=0; n<batch_size; n++) {
        for (int f=0; f<n_filters; f++) {
            for (int oh=0; oh<output_h; oh++) {
                for (int ow=0; ow<output_w; ow++) {

                    double sum = 0.0;

                    for (int c=0; c<input_c; c++) {
                        for (int kh=0; kh<filter_h; kh++) {
                            for (int kw=0; kw<filter_w; kw++) {

                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;

                                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                    double input_val = in->filters[n]->channels[c]->entries[ih][iw];
                                    double filter_val = filter->filters[f]->channels[c]->entries[kh][kw];
    
                                    sum += input_val * filter_val;
                                }
                            }
                        }
                    }

                    sum += bias->filters[f]->channels[0]->entries[0][0];
                    out->filters[n]->channels[f]->entries[oh][ow] = sum;
                }
            }
        }
    }
}

void tensor4D_rot180_into(Tensor4D* t, Tensor4D* into) {
    for (int sample=0; sample<t->n_filters; sample++) {
        for (int channel = 0; channel < t->n_channels; channel++) {
            int rows = t->n_rows;
            int cols = t->n_cols;

            for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                    into->filters[sample]->channels[channel]->entries[i][j] = 
                        t->filters[sample]->channels[channel]->entries[rows - 1 - i][cols - 1 - j];
                }
            }
        }
    }
}