#pragma once

#include "matrix.h"
#include "tensor.h"

void bias_add_to_dense_z(Matrix* restrict bias, Matrix* restrict z);
void bias_add_to_conv_z(Matrix* restrict bias, Tensor4D* restrict z);