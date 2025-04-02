#include "activation.h"

Activation* activation_new(ActivationType type, double param) {
    Activation* act = (Activation*)malloc(sizeof(Activation));
    act->type = type;
    act->activation_param = param;
    switch (type)
    {
    case SIGMOID:
        act->activation_func = sigmoid;
        act->dZ = sigmoid_dZ;
        return act;
        break;

    case RELU:
        act->activation_func = relu;
        act->dZ = relu_dZ;
        return act;
        break;
    
    case LRELU:
        act->activation_func = leaky_relu;
        act->dZ = leaky_relu_dZ;
        return act;
        break;

    case ELU:
        act->activation_func = elu;
        act->dZ = elu_dZ;
        return act;
        break;
    
    default:
        printf("Unknown activation type.");
        break;
    }
}

Matrix* apply_activation_func(Activation* activation, Matrix* z_m) {
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

void apply_activation_func_into(Activation* activation, Matrix* z_m, Matrix* into) {
    for (int i=0; i<z_m->n_rows; i++) {
        for (int j=0; j<z_m->n_cols; j++) {
            double z = z_m->entries[i][j];
            double param = activation->activation_param;
            into->entries[i][j] = activation->activation_func(z, param);
        }
    }
}

Matrix* apply_activation_dZ(Activation* activation, Matrix* z_m) {
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

void apply_activation_dZ_into(Activation* activation, Matrix* z_m, Matrix* into) {
    for (int i=0; i<z_m->n_rows; i++) {
        for (int j=0; j<z_m->n_cols; j++) {
            double z = z_m->entries[i][j];
            double param = activation->activation_param;
            into->entries[i][j] = activation->dZ(z, param);
        }
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