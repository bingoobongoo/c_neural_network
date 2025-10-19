#include "bias.h"

void bias_add_to_dense_z(Matrix* bias, Matrix* z) {
    #ifdef DEBUG

    if (bias->n_cols != z->n_cols || bias->n_rows != 1) {
        printf("(bias_add_to_dense_z) Bias is of wrong shape:\n");
        printf("Bias dimension: "); matrix_print_dimensions(bias);
        printf("Weights dimension: "); matrix_print_dimensions(z);
        exit(1);
    }

    #endif

    for (int i=0; i<z->n_cols; i++) {
        nn_float b = matrix_get(bias, 0, i);
        for (int j=0; j<z->n_rows; j++) {
            nn_float x = matrix_get(z, j, i);
            matrix_assign(
                z,
                j,
                i,
                x+b
            );
        }
    }
}

void bias_add_to_conv_z(Matrix* bias, Tensor4D* z) {
    #ifdef DEBUG

    // z->n_channels = filter->n_filters (filter is weights for cnn)
    if (bias->n_cols != z->n_channels || bias->n_rows != 1) {
        printf("(bias_add_to_dense_z) Bias is of wrong shape:\n");
        printf("Bias dimension: "); matrix_print_dimensions(bias);
        printf("Weights dimension: "); tensor4D_print_shape(z);
        exit(1);
    }

    #endif

    for (int i=0; i<z->n_channels; i++) {
        nn_float b = matrix_get(bias, 0, i);
        for (int j=0; j<z->n_filters; j++) {
            Matrix* m = z->filters[j]->channels[i];
            for (int k=0; k<z->n_rows; k++) {
                for (int l=0; l<z->n_cols; l++) {
                    nn_float x = matrix_get(m, k, l);
                    matrix_assign(
                        m,
                        k,
                        l,
                        x+b
                    );
                }
            }
        }
    }
}