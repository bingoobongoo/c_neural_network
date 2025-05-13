#include "network.h"
#include "load_data.h"
#include "preprocessing.h"

int main() {
    openblas_set_num_threads(4);
    srand(time(NULL));
    Matrix* input_m = matrix_load("input.mat");
    Matrix* kernel_m = matrix_load("kernel.mat");

    Tensor3D* input = tensor3D_new(3, 3, 2);
    Tensor4D* kernel = tensor4D_new(2, 2, 2, 2);

    matrix_into_tensor3D(input_m, input, false);
    matrix_into_tensor4D(kernel_m, kernel);

    // im2col test

    int stride = 1;
    int num_patches = pow(4, 2);
    Matrix* input_im2col = matrix_new(
        num_patches,
        kernel->n_rows*kernel->n_cols*kernel->n_channels
    );
    Matrix* kernel_im2col = matrix_new(
        kernel->n_rows*kernel->n_cols*kernel->n_channels,
        kernel->n_filters
    );
    Matrix* im2col_dot = matrix_new(
        num_patches,
        kernel->n_filters
    );
    Tensor3D* output = tensor3D_new(4, 4, kernel->n_filters);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int n=0; n<10000; n++) {
        for (int i=0; i<kernel->n_filters; i++) {
            for (int j=0; j<kernel->n_channels; j++) {
                matrix_convolve_into(
                    input->channels[j],
                    kernel->filters[i]->channels[j],
                    output->channels[j],
                    1,
                    FULL
                );
            }
        }
    }
    gettimeofday(&end, NULL);
    double conv_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Conv time taken: %.3f seconds\n", conv_time);

    // input_into_im2col(input, kernel, stride, FULL, input_im2col);
    // matrix_print(input_im2col);


    gettimeofday(&start, NULL);
    kernel_into_im2col(kernel, true, kernel_im2col);
    for (int n=0; n<10000; n++) {
        input_into_im2col(input, kernel, stride, FULL, input_im2col);
        im2col_correlate(input_im2col, kernel_im2col, im2col_dot, output);
    }    
    gettimeofday(&end, NULL);
    double im2col_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("im2col time taken: %.3f seconds\n", im2col_time);

    matrix_print(im2col_dot);
}