#include "tensor.h"

Tensor3D* tensor3D_new(int n_rows, int n_cols, int n_channels) {
    Tensor3D* t = (Tensor3D*)malloc(sizeof(Tensor3D));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->channels = (Matrix**)malloc(n_channels * sizeof(Matrix*));
    t->entries = (nn_float*)malloc(n_rows * n_cols * n_channels * sizeof(nn_float));
    t->view = false;
    for (int i=0; i<n_channels; i++) {
        t->channels[i] = matrix_view_new(n_rows, n_cols, t->entries + i*n_rows*n_cols);
        matrix_zero(t->channels[i]);
    }

    return t;
}

Tensor3D* tensor3D_view_new(int n_rows, int n_cols, int n_channels, nn_float* entries) {
    Tensor3D* t = (Tensor3D*)malloc(sizeof(Tensor3D));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->channels = (Matrix**)malloc(n_channels * sizeof(Matrix*));
    t->entries = entries;
    t->view = true;
    for (int i=0; i<n_channels; i++) {
        t->channels[i] = matrix_view_new(n_rows, n_cols, t->entries + i*n_rows*n_cols);
        matrix_zero(t->channels[i]);
    }

    return t;
}

void tensor3D_free(Tensor3D* t) {
    if (t == NULL) return;

    for (int i=0; i<t->n_channels; i++) {
        matrix_free(t->channels[i]);
    }

    free(t->channels);
    t->channels = NULL;

    if (!t->view) free(t->entries);
    t->entries = NULL;

    free(t);
}

void tensor3D_copy_into(Tensor3D* t, Tensor3D* into) {
    for (int i=0; i<t->n_channels; i++) {
        matrix_copy_into(t->channels[i], into->channels[i]);
    }
}

void tensor3D_sum_element_wise_into(Tensor3D* t, Matrix* into) {
    matrix_zero(into);
    for (int c=0; c<t->n_channels; c++) {
        for (int i=0; i<t->n_rows; i++) {
            for (int j=0; j<t->n_cols; j++) {
                nn_float sum = matrix_get(into, i, j);
                matrix_assign(into, i, j, sum + matrix_get(t->channels[c], i, j));
            }
        }
    }
}

void tensor3D_acc_correlate_into(Tensor3D* input, Tensor3D* kernel, Matrix* into, int stride, CorrelationType type) {
    #ifdef MULTI_THREADING
    #pragma omp parallel for schedule(static)
    #endif
    for (int c=0; c<input->n_channels; c++) {
        matrix_acc_correlate_into(
            input->channels[c],
            kernel->channels[c],
            into,
            stride,
            type
        );
    }
}

void tensor4D_fill(Tensor4D* t, nn_float num) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_fill(t->filters[i]->channels[j], num);
        }
    }
}

void tensor4D_zero(Tensor4D* t) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_zero(t->filters[i]->channels[j]);
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

nn_float tensor4D_sum(Tensor4D* t) {
    nn_float sum = (nn_float)0.0;
    for (int f=0; f<t->n_filters; f++) {
        for (int c=0; c<t->n_channels; c++) {
            sum += matrix_sum(t->filters[f]->channels[c]);
        }
    }

    return sum;
}

nn_float tensor4D_max(Tensor4D* t) {
    nn_float max = matrix_get(t->filters[0]->channels[0], 0, 0);
    for (int f=0; f<t->n_filters; f++) {
        for (int c=0; c<t->n_channels; c++) {
            nn_float x = matrix_max(t->filters[f]->channels[c]);
            if (x > max) max = x;
        }
    }

    return max;
}

nn_float tensor4D_min(Tensor4D* t) {
    nn_float min = matrix_get(t->filters[0]->channels[0], 0, 0);
    for (int f=0; f<t->n_filters; f++) {
        for (int c=0; c<t->n_channels; c++) {
            nn_float x = matrix_max(t->filters[f]->channels[c]);
            if (x < min) min = x;
        }
    }

    return min;
}

nn_float tensor4D_average(Tensor4D* t) {
    return tensor4D_sum(t) / (t->n_filters * t->n_channels * t->n_rows * t->n_cols);
}

Tensor4D* tensor4D_new(int n_rows, int n_cols, int n_channels, int n_filters) {
    Tensor4D* t = (Tensor4D*)malloc(sizeof(Tensor4D));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->n_filters = n_filters;
    t->filters = (Tensor3D**)malloc(n_filters * sizeof(Tensor3D*));
    t->entries = (nn_float*)malloc(n_rows * n_cols * n_channels * n_filters * sizeof(nn_float));
    for (int i=0; i<n_filters; i++) {
        t->filters[i] = tensor3D_view_new(
            n_rows, 
            n_cols, 
            n_channels,
            t->entries + i*n_rows*n_cols*n_channels
        );
    }

    return t;
}

void tensor4D_free(Tensor4D* t) {
    if (t == NULL) return;
    
    for (int i=0; i<t->n_filters; i++) {
        tensor3D_free(t->filters[i]);
    }
    free(t->filters);
    t->filters = NULL;

    free(t->entries);
    t->entries = NULL;

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

void tensor4D_flip_into(Tensor4D* t, Tensor4D* flipped) {
    for (int i=0; i<t->n_filters; i++) {
        for (int j=0; j<t->n_channels; j++) {
            matrix_flip_into(
                t->filters[i]->channels[j],
                flipped->filters[i]->channels[j]
            );
        }
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

void delta_into_im2col_fwise(Tensor4D* delta, int filter_idx, Matrix* im2col) {
    int delta_c = delta->n_channels;
    int out_h = delta->n_rows;
    int out_w = delta->n_cols;
    int hw = out_h * out_w;

    nn_float* srcs[delta_c];
    for (int c=0; c<delta_c; c++) {
        srcs[c] = delta->filters[filter_idx]->channels[c]->entries;
    }

    for (int i=0; i<hw; i++) {
        nn_float* row = im2col->entries + i*delta_c;
        
        for (int c=0; c<delta_c; c++) {
            row[c] = srcs[c][i];
        }
    }
}

void input_into_im2col_fwise(Tensor4D* input, int filter_idx, Tensor4D* kernel, int stride, int padding, Matrix* im2col) {
    int in_h = input->n_rows;
    int in_w = input->n_cols;
    int in_c = input->n_channels;

    int ker_h = kernel->n_rows;
    int ker_w = kernel->n_cols;

    int out_h = (in_h + 2*padding - ker_h) / stride + 1;
    int out_w = (in_w + 2*padding - ker_w) / stride + 1;

    for (int i=0; i<out_h; i++) {
        int is = i*stride - padding;
        for (int j=0; j<out_w; j++) {
            int js = j*stride - padding;

            nn_float* im2col_row = im2col->entries + (i*out_w + j)*im2col->n_cols;
            int col = 0;

            for (int k=0; k<in_c; k++) {
                Matrix* cm = input->filters[filter_idx]->channels[k];
                if (is>=0 && js>=0 && is+ker_h<=in_h && js+ker_w<=in_w) {
                    for (int l=0; l<ker_h; l++) {
                        nn_float* src = cm->entries + (is+l)*cm->n_cols + js;
                        nn_float* dst = im2col_row + col + l*ker_w;
                        memcpy(dst, src, ker_w*sizeof(nn_float));
                    }
                    col += ker_h*ker_w;
                }
                else {
                    for (int l=0; l<ker_h; l++) {
                        int isl = is + l;
                        bool cond = (isl>=0 && isl<in_h);
                        for (int m=0; m<ker_w; m++, col++) {
                            int jsm = js + m;
                            if (cond && jsm>=0 && jsm<in_w)
                                im2col_row[col] = cm->entries[isl*cm->n_cols + jsm];
                            else
                                im2col_row[col] = (nn_float)0.0;
                        }
                    }
                }
            }
        }
    }
}

size_t tensor3D_get_sizeof_mem_allocated(Tensor3D* t) {
    size_t size = 0;
    if (t == NULL) return size;

    size += sizeof(*t);
    for (int i=0; i<t->n_channels; i++) {
        size += matrix_get_sizeof_mem_allocated(t->channels[i]);
    }

    if (!t->view)
        size += t->n_rows * t->n_cols * t->n_channels * sizeof(nn_float);

    return size;
}

size_t tensor4D_get_sizeof_mem_allocated(Tensor4D* t) {
    size_t size = 0;
    if (t == NULL) return size;

    size += sizeof(*t);
    for (int i=0; i<t->n_filters; i++) {
        size += tensor3D_get_sizeof_mem_allocated(t->filters[i]);
    }

    size += t->n_rows * t->n_cols * t->n_channels * t->n_filters * sizeof(nn_float);

    return size;
}

size_t tensor4D_uint16_get_sizeof_mem_allocated(Tensor4D_uint16* t) {
    size_t size = 0;
    if (t == NULL) return size;

    size += sizeof(*t);
    t->n_filters * t->n_channels * t->n_rows * t->n_cols * sizeof(uint16_t);

    return size;
}

Tensor3D_uint16* tensor3D_uint16_new(int n_rows, int n_cols, int n_channels) {
    Tensor3D_uint16* t = (Tensor3D_uint16*)malloc(sizeof(Tensor3D_uint16));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->channels = (Matrix_uint16**)malloc(n_channels * sizeof(Matrix_uint16*));
    for (int i=0; i<n_channels; i++) {
        t->channels[i] = matrix_uint16_new(n_rows, n_cols);
        matrix_uint16_fill(t->channels[i], (uint16_t)0);
    }

    return t;  
}

void tensor3D_uint16_free(Tensor3D_uint16* t) {
    for (int i=0; i<t->n_channels; i++) {
        matrix_uint16_free(t->channels[i]);
    }
    free(t->channels);
    free(t);
}

Tensor4D_uint16* tensor4D_uint16_new(int n_rows, int n_cols, int n_channels, int n_filters) {
    Tensor4D_uint16* t = (Tensor4D_uint16*)malloc(sizeof(Tensor4D_uint16));
    t->n_rows = n_rows;
    t->n_cols = n_cols;
    t->n_channels = n_channels;
    t->n_filters = n_filters;
    t->filters = (Tensor3D_uint16**)malloc(n_filters * sizeof(Tensor3D_uint16*));
    for (int i=0; i<n_filters; i++) {
        t->filters[i] = tensor3D_uint16_new(n_rows, n_cols, n_channels);
    }

    return t;
}

void tensor4D_uint16_free(Tensor4D_uint16* t) {
    for (int i=0; i<t->n_filters; i++) {
        tensor3D_uint16_free(t->filters[i]);
    }
    free(t->filters);
    free(t);
}