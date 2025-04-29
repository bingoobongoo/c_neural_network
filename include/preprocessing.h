#pragma once

#include "matrix.h"
#include "tensor.h"

Matrix* one_hot_encode(Matrix* column, int n_classes);
void normalize(Matrix* m);
void renormalize(Matrix* m, int original_min, int original_max);
void shuffle_matrix_inplace(Matrix* feature_m, Matrix* label_m);
void shuffle_tensor4D_inplace(Tensor4D* feature_t, Matrix* label_m);