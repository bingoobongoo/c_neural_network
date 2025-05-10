#include "network.h"
#include "load_data.h"
#include "preprocessing.h"

int main() {
    srand(time(NULL));
    Matrix* mat1 = matrix_load("mat1.mat");
    Matrix* mat2 = matrix_load("mat2.mat");
    Matrix* mat3 = matrix_new(4,4);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i=0; i<100000000; i++) {
        matrix_multiply_into(mat1, mat2, mat3);
    }
    gettimeofday(&end, NULL);
    double time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("time: %.5f seconds\n", time);

    matrix_print(mat1);
    printf("\n");
    matrix_print(mat2);
    printf("\n");
    matrix_print(mat3);

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(mat3);
}