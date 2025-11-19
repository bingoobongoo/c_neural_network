#include "scalar.h"

nn_float powi(nn_float x, int y) {
    if (y == 0) return 1;

    nn_float num = (y > 0) ? x : (nn_float)1.0/x;
    nn_float sum = (nn_float)1.0;

    int n_iter = (y > 0) ? y : -y;
    for (int i=0; i<n_iter; i++) {
        sum *= num;
    }

    return sum;
}