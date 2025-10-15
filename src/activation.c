#include "activation.h"

Activation* activation_new(ActivationType type, nn_float param) {
    Activation* act = (Activation*)malloc(sizeof(Activation));
    act->type = type;
    act->y_true_batch = NULL;
    act->activation_param = param;
    switch (type)
    {
    case SIGMOID:
        act->activation_func = sigmoid;
        act->dZ = sigmoid_dZ;
        act->name = "Sigmoid";
        return act;
        break;

    case RELU:
        act->activation_func = relu;
        act->dZ = relu_dZ;
        act->name = "RELU";
        return act;
        break;
    
    case LRELU:
        act->activation_func = leaky_relu;
        act->dZ = leaky_relu_dZ;
        act->name = "Leaky RELU";
        return act;
        break;

    case ELU:
        act->activation_func = elu;
        act->dZ = elu_dZ;
        act->name = "ELU";
        return act;
        break;
    
    case SOFTMAX:
        act->activation_func = NULL;
        act->dZ = NULL;
        act->name = "Softmax";
        return act;
        break;
    
    default:
        printf("Unknown activation type.");
        break;
    }
}

Matrix* apply_activation_func(Activation* activation, Matrix* z_m) {
    if (activation->type != SOFTMAX) {
        Matrix* a = matrix_new(z_m->n_rows, z_m->n_cols);
        for (int i=0; i<z_m->n_rows; i++) {
            for (int j=0; j<z_m->n_cols; j++) {
                nn_float z = matrix_get(z_m, i, j);
                nn_float param = activation->activation_param;
                matrix_assign(a, i, j, activation->activation_func(z, param));
            }
        }
    
        return a;
    }
    else if (activation->type == SOFTMAX) {
        Matrix* a = matrix_new(z_m->n_rows, z_m->n_cols);
        for (int i=0; i<z_m->n_rows; i++) {
            nn_float max_z = matrix_get(z_m, i, 0);
            for (int j=0; j<z_m->n_cols; j++) {
                nn_float z = matrix_get(z_m, i, j);
                if (z > max_z)
                    max_z = z;
            }

            nn_float sum_z = (nn_float)0.0;
            for (int j=0; j<a->n_cols; j++) {
                #ifdef SINGLE_PRECISION
                nn_float e = expf(matrix_get(z_m, i, j) - max_z);
                #endif
                #ifdef DOUBLE_PRECISION
                nn_float e = exp(matrix_get(z_m, i, j) - max_z);
                #endif
                matrix_assign(a, i, j, e);
                sum_z += e;
            }

            for (int j=0; j<a->n_cols; j++) {
                matrix_assign(a, i, j, matrix_get(a, i, j) / sum_z);
            }
        }

        return a;
    }
}

void apply_activation_func_into(Activation* activation, Matrix* z_m, Matrix* into) {
    if (activation->type != SOFTMAX) {
        for (int i=0; i<z_m->n_rows; i++) {
            for (int j=0; j<z_m->n_cols; j++) {
                nn_float z = matrix_get(z_m, i, j);
                nn_float param = activation->activation_param;
                matrix_assign(into, i, j, activation->activation_func(z, param));
            }
        }
    }
    else if (activation->type == SOFTMAX) {
        for (int i=0; i<z_m->n_rows; i++) {
            nn_float max_z = matrix_get(z_m, i, 0);
            for (int j=0; j<z_m->n_cols; j++) {
                nn_float z = matrix_get(z_m, i, j);
                if (z > max_z)
                    max_z = z;
            }

            nn_float sum_z = (nn_float)0.0;
            for (int j=0; j<into->n_cols; j++) {
                #ifdef SINGLE_PRECISION
                nn_float e = expf(matrix_get(z_m, i, j) - max_z);
                #endif
                #ifdef DOUBLE_PRECISION
                nn_float e = exp(matrix_get(z_m, i, j) - max_z);
                #endif
                matrix_assign(into, i, j, e);
                sum_z += e;
            }

            for (int j=0; j<into->n_cols; j++) {
                matrix_assign(into, i, j, matrix_get(into, i, j) / sum_z);
            }
        }
    } 
}

Matrix* apply_activation_dZ(Activation* activation, Matrix* z_m) {
    if (activation->type != SOFTMAX) {
        Matrix* dZ = matrix_new(z_m->n_rows, z_m->n_cols);
        for (int i=0; i<z_m->n_rows; i++) {
            for (int j=0; j<z_m->n_cols; j++) {
                nn_float z = matrix_get(z_m, i, j);
                nn_float param = activation->activation_param;
                matrix_assign(dZ, i, j, activation->dZ(z, param));
            }
        }
    
        return dZ;
    }
    else if (activation->type == SOFTMAX) {
        Matrix* dZ = apply_activation_func(activation, z_m);
        matrix_subtract_into(dZ, activation->y_true_batch->data.matrix, dZ);

        return dZ;
    }
}

void apply_activation_dZ_into(Activation* activation, Matrix* z_m, Matrix* into) {
    if (activation->type != SOFTMAX) {
        for (int i=0; i<z_m->n_rows; i++) {
            for (int j=0; j<z_m->n_cols; j++) {
                nn_float z = matrix_get(z_m, i, j);
                nn_float param = activation->activation_param;
                matrix_assign(into, i, j, activation->dZ(z, param));
            }
        }
    }
    else if (activation->type == SOFTMAX) {
        apply_activation_func_into(activation, z_m, into);
        matrix_subtract_into(into, activation->y_true_batch->data.matrix, into);
    }
}

nn_float sigmoid(nn_float z, nn_float param) {
    #ifdef SINGLE_PRECISION
    return (nn_float)1.0/((nn_float)1.0 + expf(-z));
    #endif
    #ifdef DOUBLE_PRECISION
    return (nn_float)1.0/((nn_float)1.0 + exp(-z));
    #endif
}

nn_float sigmoid_dZ(nn_float z, nn_float param) {
    return sigmoid(z, param) * ((nn_float)1.0 - sigmoid(z, param));
}

nn_float relu(nn_float z, nn_float param) {
    if (z <= (nn_float)0.0) {
        return (nn_float)0.0;
    }

    return z;
}

nn_float relu_dZ(nn_float z, nn_float param) {
    if (z <= (nn_float)0.0) {
        return (nn_float)0.0;
    }

    return (nn_float)1.0;
}

nn_float leaky_relu(nn_float z, nn_float param) {
    if (z <= (nn_float)0.0) {
        return param * z;
    }

    return z;
}

nn_float leaky_relu_dZ(nn_float z, nn_float param) {
    if (z <= (nn_float)0.0) {
        return param;
    }

    return (nn_float)1.0;
}

nn_float elu(nn_float z, nn_float param) {
    if (z <= (nn_float)0.0) {
        #ifdef SINGLE_PRECISION
        return param * (expf(z) - (nn_float)1.0);
        #endif
        #ifdef DOUBLE_PRECISION
        return param * (exp(z) - (nn_float)1.0);
        #endif
    }

    return z;
}

nn_float elu_dZ(nn_float z, nn_float param) {
    if (z <= (nn_float)0.0) {
        return elu(z, param) + param;
    }

    return (nn_float)1.0;
}