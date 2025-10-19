#pragma once

#include "matrix.h"
#include "tensor.h"

void bias_add_to_dense_z(Matrix* bias, Matrix* z);
void bias_add_to_conv_z(Matrix* bias, Tensor4D* z);