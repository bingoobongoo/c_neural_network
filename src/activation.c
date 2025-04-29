#include "activation.h"

Activation* activation_new(ActivationType type, double param) {
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
                double z = z_m->entries[i][j];
                double param = activation->activation_param;
                a->entries[i][j] = activation->activation_func(z, param);
            }
        }
    
        return a;
    }
    else if (activation->type == SOFTMAX) {
        Matrix* a = matrix_new(z_m->n_rows, z_m->n_cols);
        for (int i=0; i<z_m->n_rows; i++) {
            double max_z = z_m->entries[i][0];
            for (int j=0; j<z_m->n_cols; j++) {
                if (z_m->entries[i][j] > max_z)
                max_z = z_m->entries[i][j];
            }

            double sum_z = 0.0;
            for (int j=0; j<a->n_cols; j++) {
                a->entries[i][j] = exp(z_m->entries[i][j] - max_z);
                sum_z += a->entries[i][j];
            }

            for (int j=0; j<a->n_cols; j++) {
                a->entries[i][j] /= sum_z;
            }
        }

        return a;
    }
}

void apply_activation_func_into(Activation* activation, Matrix* z_m, Matrix* into) {
    if (activation->type != SOFTMAX) {
        for (int i=0; i<z_m->n_rows; i++) {
            for (int j=0; j<z_m->n_cols; j++) {
                double z = z_m->entries[i][j];
                double param = activation->activation_param;
                into->entries[i][j] = activation->activation_func(z, param);
            }
        }
    }
    else if (activation->type == SOFTMAX) {
        for (int i=0; i<z_m->n_rows; i++) {
            double max_z = z_m->entries[i][0];
            for (int j=0; j<z_m->n_cols; j++) {
                if (z_m->entries[i][j] > max_z)
                max_z = z_m->entries[i][j];
            }

            double sum_z = 0.0;
            for (int j=0; j<into->n_cols; j++) {
                into->entries[i][j] = exp(z_m->entries[i][j] - max_z);
                sum_z += into->entries[i][j];
            }

            for (int j=0; j<into->n_cols; j++) {
                into->entries[i][j] /= sum_z;
            }
        }
    } 
}

Matrix* apply_activation_dZ(Activation* activation, Matrix* z_m) {
    if (activation->type != SOFTMAX) {
        Matrix* dZ = matrix_new(z_m->n_rows, z_m->n_cols);
        for (int i=0; i<z_m->n_rows; i++) {
            for (int j=0; j<z_m->n_cols; j++) {
                double z = z_m->entries[i][j];
                double param = activation->activation_param;
                dZ->entries[i][j] = activation->dZ(z, param);
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
                double z = z_m->entries[i][j];
                double param = activation->activation_param;
                into->entries[i][j] = activation->dZ(z, param);
            }
        }
    }
    else if (activation->type == SOFTMAX) {
        apply_activation_func_into(activation, z_m, into);
        matrix_subtract_into(into, activation->y_true_batch->data.matrix, into);
    }
}

double sigmoid(double z, double param) {
    return 1.0/(1.0 + exp(-z));
}

double sigmoid_dZ(double z, double param) {
    return sigmoid(z, param) * (1.0 - sigmoid(z, param));
}

double relu(double z, double param) {
    if (z <= 0.0) {
        return 0.0;
    }

    return z;
}

double relu_dZ(double z, double param) {
    if (z <= 0) {
        return 0.0;
    }

    return 1.0;
}

double leaky_relu(double z, double param) {
    if (z <= 0.0) {
        return param * z;
    }

    return z;
}

double leaky_relu_dZ(double z, double param) {
    if (z <= 0) {
        return param;
    }

    return 1.0;
}

double elu(double z, double param) {
    if (z <= 0.0) {
        return param * (exp(z) - 1.0);
    }

    return z;
}

double elu_dZ(double z, double param) {
    if (z <= 0.0) {
        return elu(z, param) + param;
    }

    return 1.0;
}