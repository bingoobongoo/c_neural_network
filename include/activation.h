#pragma once

#include "matrix.h"

typedef enum {
    SIGMOID,
    RELU,
    LRELU,
    ELU
} ActivationType;

typedef struct {
    ActivationType type;
    double(*activation_func)(double, double);
    double(*dZ)(double, double);
    double activation_param;
} Activation;

Activation* activation_new(ActivationType type, double param);

Matrix* apply_activation_func(Activation* activation, Matrix* z_m);
void apply_activation_func_into(Activation* activation, Matrix* z_m, Matrix* into);
Matrix* apply_activation_dZ(Activation* activation, Matrix* z_m);
void apply_activation_dZ_into(Activation* activation, Matrix* z_m, Matrix* into);

double sigmoid(double z, double param);
double sigmoid_dZ(double z, double param);

double relu(double z, double param);
double relu_dZ(double z, double param);

double leaky_relu(double z, double param);
double leaky_relu_dZ(double z, double param);

double elu(double z, double param);
double elu_dZ(double z, double param);