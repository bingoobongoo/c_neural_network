#include "network.h"
#include "load_data.h"
#include "preprocessing.h"

int main() {
    srand(time(NULL));
    Matrix* input_m = matrix_load("input.mat");
    Matrix* kernel_m = matrix_load("kernel.mat");

    Tensor3D* input = tensor3D_new(3, 3, 2);
    Tensor4D* kernel = tensor4D_new(2, 2, 2, 2);

    matrix_into_tensor3D(input_m, input, false);
    matrix_into_tensor4D(kernel_m, kernel);

    // im2col test

    int stride = 1;
    int num_patches = pow((input->n_rows - kernel->n_rows)/stride + 1, 2);
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
    Tensor3D* output = tensor3D_new(2, 2, kernel->n_filters);

    input_into_im2col(input, kernel, stride, input_im2col);
    kernel_into_im2col(kernel, kernel_im2col);

    // matrix_print(input_im2col);
    // matrix_print(kernel_im2col);

    im2col_correlate(input_im2col, kernel_im2col, im2col_dot, output);

    matrix_print(output->channels[1]);
    
}