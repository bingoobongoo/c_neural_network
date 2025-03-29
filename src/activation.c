#include "activation.h"

Activation* activation_new(ActivationType type, double param) {
    Activation* act = (Activation*)malloc(sizeof(Activation));
    act->type = type;
    act->activation_param = param;
    switch (type)
    {
    case SIGMOID:
        act->activation_func = sigmoid_activation;
        act->derivative = sigmoid_derivative;
        return act;
        break;

    case RELU:
        act->activation_func = relu_activation;
        act->derivative = relu_derivative;
        return act;
        break;
    
    case LRELU:
        act->activation_func = leaky_relu_activation;
        act->derivative = leaky_relu_derivative;
        return act;
        break;

    case ELU:
        act->activation_func = elu_activation;
        act->derivative = elu_derivative;
        return act;
        break;
    
    default:
        printf("Unknown activation type.");
        break;
    }
    
}

double sigmoid_activation(double x, double param) {
    return 1.0/(1.0 + exp(-x));
}

double sigmoid_derivative(double x, double param) {
    return sigmoid_activation(x, param) * (1.0 - sigmoid_activation(x, param));
}

double relu_activation(double x, double param) {
    if (x <= 0.0) {
        return 0.0;
    }

    return x;
}

double relu_derivative(double x, double param) {
    if (x <= 0) {
        return 0.0;
    }

    return 1.0;
}

double leaky_relu_activation(double x, double param) {
    if (param * x >= x) {
        return param * x;
    }

    return x;
}

double leaky_relu_derivative(double x, double param) {
    if (x <= 0) {
        return param;
    }

    return 1.0;
}

double elu_activation(double x, double param) {
    if (x <= 0.0) {
        return param * (exp(x) - 1.0);
    }

    return x;
}

double elu_derivative(double x, double param) {
    if (x <= 0.0) {
        return elu_activation(x, param) + param;
    }

    return 1.0;
}