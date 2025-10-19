#include "tensor.h"

Tensor3D* tensor3D_new(int n_rows, int n_cols, int n_channels) {
    Tensor3D* t = (Tensor3D*)malloc(sizeof(Tensor3D));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->channels = (Matrix**)malloc(n_channels * sizeof(Matrix*));
    for (int i=0; i<n_channels; i++) {
        t->channels[i] = matrix_new(n_rows, n_cols);
        matrix_fill(t->channels[i], (nn_float)0.0);
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
    matrix_fill(into, (nn_float)0.0);
    for (int c=0; c<t->n_channels; c++) {
        for (int i=0; i<t->n_rows; i++) {
            for (int j=0; j<t->n_cols; j++) {
                nn_float sum = matrix_get(into, i, j);
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

void tensor4D_fill(Tensor4D* t, nn_float num) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_fill(t->filters[i]->channels[j], num);
        }
    }
}

void tensor4D_fill_normal_distribution(Tensor4D* t, nn_float mean, nn_float std_deviation) {
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

void tensor4D_apply_inplace(nn_float (*func)(nn_float), Tensor4D* t) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_apply_inplace(
                func,
                t->filters[i]->channels[j]
            );
        }
    }
}

void tensor4D_scale_inplace(nn_float scalar, Tensor4D* t) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_scale_inplace(
                scalar,
                t->filters[i]->channels[j]
            );
        }
    }
}

void tensor4D_scale_into(nn_float scalar, Tensor4D* t, Tensor4D* into) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_scale_into(
                scalar,
                t->filters[i]->channels[j],
                into->filters[i]->channels[j]
            );
        }
    }    
}

void tensor4D_add_into(Tensor4D* t1, Tensor4D* t2, Tensor4D* into) {
    for (int i=0; i<t1->n_filters; i++) {
        for (int j=0; j<t1->n_channels; j++) {
            matrix_add_into(
                t1->filters[i]->channels[j],
                t2->filters[i]->channels[j],
                into->filters[i]->channels[j]
            );
        }
    }    
}

void tensor4D_add_scalar_into(nn_float scalar, Tensor4D* t, Tensor4D* into) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_add_scalar_into(
                scalar,
                t->filters[i]->channels[j],
                into->filters[i]->channels[j]
            );
        }
    }  
}

void tensor4D_subtract_into(Tensor4D* t1, Tensor4D* t2, Tensor4D* into) {
    for (int i=0; i<t1->n_filters; i++) {
        for (int j=0; j<t1->n_channels; j++) {
            matrix_subtract_into(
                t1->filters[i]->channels[j],
                t2->filters[i]->channels[j],
                into->filters[i]->channels[j]
            );
        }
    }    
}

void tensor4D_multiply_into(Tensor4D* t1, Tensor4D* t2, Tensor4D* into) {
    for (int i=0; i<t1->n_filters; i++) {
        for (int j=0; j<t1->n_channels; j++) {
            matrix_multiply_into(
                t1->filters[i]->channels[j],
                t2->filters[i]->channels[j],
                into->filters[i]->channels[j]
            );
        }
    }   
}

void tensor4D_divide_into(Tensor4D* t1, Tensor4D* t2, Tensor4D* into) {
    for (int i=0; i<t1->n_filters; i++) {
        for (int j=0; j<t1->n_channels; j++) {
            matrix_divide_into(
                t1->filters[i]->channels[j],
                t2->filters[i]->channels[j],
                into->filters[i]->channels[j]
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

void tensor4D_print_shape(Tensor4D* t) {
    printf("[%d x %d x %d x %d]\n", 
        t->n_filters,
        t->n_channels,
        t->n_rows,
        t->n_cols
    );
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

void tensor4D_into_matrix_fwise(Tensor4D* t, Matrix* m, bool transpose, bool flipped) {
    if (transpose && !flipped) {
        int jk, kk;
        for (int i=0; i<t->n_filters; i++) {
            Tensor3D* t3d = t->filters[i];
    
            for (int j=0; j<t->n_channels; j++) {
                Matrix* channel_mat = t3d->channels[j];
                jk = j*t->n_rows*t->n_cols;
                for (int k=0; k<t->n_rows; k++) {
                    kk = k*t->n_cols;
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = jk + kk + l;
                        matrix_assign(m, idx, i, matrix_get(channel_mat, k, l));
                    }
                }
            }
        }
    }
    else if (transpose && flipped) {
        int jk, kk;
        for (int i=0; i<t->n_filters; i++) {
            Tensor3D* t3d = t->filters[i];
    
            for (int j=0; j<t->n_channels; j++) {
                Matrix* channel_mat = t3d->channels[j];
                jk = j*t->n_rows*t->n_cols;
                for (int k=0; k<t->n_rows; k++) {
                    kk = k*t->n_cols;
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = jk + kk + l;
                        matrix_assign(m, idx, i, matrix_get(
                                channel_mat, 
                                t->n_rows - k - 1, 
                                t->n_cols - l - 1
                            )
                        );
                    }
                }
            }
        }
    }
    else if (!transpose && !flipped) {
        int jk, kk;
        for (int i=0; i<t->n_filters; i++) {
            Tensor3D* t3d = t->filters[i];
    
            for (int j=0; j<t->n_channels; j++) {
                Matrix* channel_mat = t3d->channels[j];
                jk = j*t->n_rows*t->n_cols;
                for (int k=0; k<t->n_rows; k++) {
                    kk = k*t->n_cols;
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = jk + kk + l;
                        matrix_assign(m, i, idx, matrix_get(channel_mat, k, l));
                    }
                }
            }
        }
    }
    else if (!transpose && flipped) {
        int jk, kk;
        for (int i=0; i<t->n_filters; i++) {
            Tensor3D* t3d = t->filters[i];
    
            for (int j=0; j<t->n_channels; j++) {
                Matrix* channel_mat = t3d->channels[j];
                jk = j*t->n_rows*t->n_cols;
                for (int k=0; k<t->n_rows; k++) {
                    kk = k*t->n_cols;
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = jk + kk + l;
                        matrix_assign(m, i, idx, matrix_get(
                                channel_mat, 
                                t->n_rows - k - 1, 
                                t->n_cols - l - 1
                            )
                        );
                    }
                }
            }
        }
    }
}

void tensor4D_into_matrix_chwise(Tensor4D* t, Matrix* m, bool transpose, bool flipped) {
    if (transpose && !flipped) {
        int jk, kk;
        for (int i=0; i<t->n_channels; i++) {
            for (int j=0; j<t->n_filters; j++) {
                Matrix* filter_mat = t->filters[j]->channels[i];
                jk = j*t->n_rows*t->n_cols;
                for (int k=0; k<t->n_rows; k++) {
                    kk = k*t->n_cols;
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = jk + kk + l;
                        matrix_assign(m, idx, i, matrix_get(filter_mat, k, l));
                    }
                }
            }
        }        
    }
    else if (transpose && flipped) {
        int jk, kk;
        for (int i=0; i<t->n_channels; i++) {
            for (int j=0; j<t->n_filters; j++) {
                Matrix* filter_mat = t->filters[j]->channels[i];
                jk = j*t->n_rows*t->n_cols;
                for (int k=0; k<t->n_rows; k++) {
                    kk = k*t->n_cols;
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = jk + kk + l;
                        matrix_assign(m, idx, i, matrix_get(
                                filter_mat, 
                                t->n_rows - k - 1, 
                                t->n_cols - l - 1
                            )
                        );
                    }
                }
            }
        }        
    }
    else if (!transpose && !flipped) {
        int jk, kk;
        for (int i=0; i<t->n_channels; i++) {
            for (int j=0; j<t->n_filters; j++) {
                Matrix* filter_mat = t->filters[j]->channels[i];
                jk = j*t->n_rows*t->n_cols;
                for (int k=0; k<t->n_rows; k++) {
                    kk = k*t->n_cols;
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = jk + kk + l;
                        matrix_assign(m, i, idx, matrix_get(filter_mat, k, l));
                    }
                }
            }
        }        
    }
    else if (!transpose && flipped) {
        int jk, kk;
        for (int i=0; i<t->n_channels; i++) {
            for (int j=0; j<t->n_filters; j++) {
                Matrix* filter_mat = t->filters[j]->channels[i];
                jk = j*t->n_rows*t->n_cols;
                for (int k=0; k<t->n_rows; k++) {
                    kk = k*t->n_cols;
                    for (int l=0; l<t->n_cols; l++) {
                        int idx = jk + kk + l;
                        matrix_assign(m, i, idx, matrix_get(
                                filter_mat, 
                                t->n_rows - k - 1, 
                                t->n_cols - l - 1
                            )
                        );
                    }
                }
            }
        }  
    }
}

void kernel_into_im2col_fwise(Tensor4D* kernel, bool flipped, Matrix* kernel_im2col) {
    tensor4D_into_matrix_fwise(kernel, kernel_im2col, true, flipped);
}

void kernel_into_im2col_chwise(Tensor4D* kernel, bool flipped, Matrix* kernel_im2col) {
    tensor4D_into_matrix_chwise(kernel, kernel_im2col, true, flipped);
}

void input_into_im2col_fwise(Tensor4D* input, int filter_idx, Tensor4D* kernel, int stride, CorrelationType corr_type,  Matrix* input_im2col) {
    switch (corr_type)
    {
    case VALID: {
        int out_h = (input->n_rows - kernel->n_rows)/stride + 1;
        int out_w = (input->n_cols - kernel->n_cols)/stride + 1;
        int ker_h = kernel->n_rows;
        int ker_w = kernel->n_cols;
        int input_h_idx, input_w_idx;
        int dest_h_idx, dest_w_idx;
        int ck, kk, is, js;

        for (int i=0; i<out_h; i++) {
            is = i*stride;
            for (int j=0; j<out_w; j++) {
                js = j*stride;
                dest_h_idx = i*out_w + j;
                for (int c=0; c<input->n_channels; c++) {
                    ck = c*ker_h*ker_w;
                    for (int k=0; k<ker_h; k++) {
                        kk = k*ker_w;
                        for (int l=0; l<ker_w; l++) {
                            matrix_assign(
                                input_im2col,
                                dest_h_idx,
                                ck + kk + l,
                                matrix_get(
                                    input->filters[filter_idx]->channels[c], 
                                    is+k, 
                                    js+l
                                )
                            );
                        }
                    }
                }
            }
        }
        break;
    }
    
    case FULL: {
        int ker_h = kernel->n_rows;
        int ker_w = kernel->n_cols;
        int out_h = (input->n_rows + ker_h - 2 + stride)/stride;
        int out_w = (input->n_cols + ker_w - 2 + stride)/stride;
        int dest_h_idx, dest_w_idx;
        int ki, lj, ck, kk, is, js;
        int k_stick_out, l_stick_out;
        int k_thresh, l_thresh;

        matrix_zero(input_im2col);

        for (int i=0; i<out_h; i++) {
            is = i*stride;
            ki = ker_h - is - 1;
            if (ki<0) ki=0;
            k_stick_out = is - input->n_rows + 1;
            if (k_stick_out<0) k_stick_out=0;
            k_thresh = ker_h - k_stick_out;
            for (int j=0; j<out_w; j++) {
                js = j*stride;
                lj = ker_w - js - 1;
                if (lj<0) lj=0;
                l_stick_out = js - input->n_cols + 1;
                if (l_stick_out<0) l_stick_out=0;
                l_thresh = ker_w - l_stick_out;
                dest_h_idx = i*out_w + j;
                for (int c=0; c<input->n_channels; c++) {
                    ck = c*ker_h*ker_w;
                    for (int k=ki; k<k_thresh; k++) {
                        kk = k*ker_w;
                        for (int l=lj; l<l_thresh; l++) {
                            matrix_assign(
                                input_im2col,
                                dest_h_idx,
                                ck + kk + l,
                                matrix_get(
                                    input->filters[filter_idx]->channels[c], 
                                    is + k - ker_h + 1, 
                                    js + l - ker_w + 1
                                )
                            );
                        }
                    }
                }
            }
        }
        break;
    }
    }
}

void input_into_im2col_chwise(Tensor4D* input, int channel_idx, Tensor4D* kernel, int stride, CorrelationType corr_type,  Matrix* input_im2col) {
    switch (corr_type)
    {
    case VALID: {
        int out_h = (input->n_rows - kernel->n_rows)/stride + 1;
        int out_w = (input->n_cols - kernel->n_cols)/stride + 1;
        int ker_h = kernel->n_rows;
        int ker_w = kernel->n_cols;
        int input_h_idx, input_w_idx;
        int dest_h_idx, dest_w_idx;
        int nk, kk, is, js;

        for (int i=0; i<out_h; i++) {
            is = i*stride;
            for (int j=0; j<out_w; j++) {
                js = j*stride;
                dest_h_idx = i*out_w + j;
                for (int n=0; n<input->n_filters; n++) {
                    nk = n*ker_h*ker_w;
                    for (int k=0; k<ker_w; k++) {
                        kk = k*ker_w;
                        for (int l=0; l<ker_w; l++) {
                            matrix_assign(
                                input_im2col,
                                dest_h_idx,
                                nk + kk + l,
                                matrix_get(
                                    input->filters[n]->channels[channel_idx],
                                    is+k,
                                    js+l
                                )
                            );
                        }
                    }
                }
            }
        }
        break;
    }
    
    case FULL: {
        int ker_h = kernel->n_rows;
        int ker_w = kernel->n_cols;
        int out_h = (input->n_rows + ker_h - 2 + stride)/stride;
        int out_w = (input->n_cols + ker_w - 2 + stride)/stride;
        int dest_h_idx, dest_w_idx;
        int ki, lj, nk, kk, is, js;
        int k_stick_out, l_stick_out;
        int k_thresh, l_thresh;

        matrix_zero(input_im2col);

        for (int i=0; i<out_h; i++) {
            is = i*stride;
            ki = ker_h - is - 1;
            if (ki<0) ki=0;
            k_stick_out = is - input->n_rows + 1;
            if (k_stick_out<0) k_stick_out=0;
            k_thresh = ker_h - k_stick_out;
            for (int j=0; j<out_w; j++) {
                js = j*stride;
                lj = ker_w - js - 1;
                if (lj<0) lj=0;
                l_stick_out = js - input->n_cols + 1;
                if (l_stick_out<0) l_stick_out=0;
                l_thresh = ker_w - l_stick_out;
                dest_h_idx = i*out_w + j;
                for (int n=0; n<input->n_filters; n++) {
                    nk = n*ker_h*ker_w;
                    for (int k=ki; k<k_thresh; k++) {
                        kk = k*ker_w;
                        for (int l=lj; l<l_thresh; l++) {
                            matrix_assign(
                                input_im2col,
                                dest_h_idx,
                                nk + kk + l,
                                matrix_get(
                                    input->filters[n]->channels[channel_idx], 
                                    is + k - ker_h + 1, 
                                    js + l - ker_w + 1
                                )
                            );
                        }
                    }
                }
            }
        }

        break;
    }
    }
}