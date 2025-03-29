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
    double(*derivative)(double, double);
    double activation_param;
} Activation;

Activation* activation_new(ActivationType type, double param);

double sigmoid_activation(double x, double param);
double sigmoid_derivative(double x, double param);

double relu_activation(double x, double param);
double relu_derivative(double x, double param);

double leaky_relu_activation(double x, double param);
double leaky_relu_derivative(double x, double param);

double elu_activation(double x, double param);
double elu_derivative(double x, double param);