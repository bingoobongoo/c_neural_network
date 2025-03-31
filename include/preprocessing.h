#pragma once

#include "matrix.h"

Matrix* one_hot_encode(Matrix* column, int n_classes);
void normalize(Matrix* m);
void renormalize(Matrix* m, int original_min, int original_max);
void shuffle_data_inplace(Matrix* feature_m, Matrix* label_m);