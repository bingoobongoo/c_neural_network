#include "network.h"

NeuralNet* neural_net_new(ActivationType activation_type, double activation_param, int batch_size) {
    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    net->n_in_layers = 0;
    net->n_ou_layers = 0;
    net->n_de_layers = 0;
    net->activation = activation_new(activation_type, activation_param);
    net->batch_size = batch_size;
    net->layers = (Layer**)malloc(100 * sizeof(Layer*));
    net->compiled = false;

    return net;
}

void neural_net_free(NeuralNet* net) {
    int n_layers = net->n_in_layers + net->n_ou_layers + net->n_de_layers;
    for (int i=0; i<n_layers; i++) {
        layer_free(net->layers[i]);
    }
    
    free(net->layers);
    free(net->activation);
    free(net);
}

void neural_net_compile(NeuralNet* net) {
    // 1. link layers
    // 2. initialize activation
    // 3. initialize weight and bias matrices
    neural_net_link_layers(net);
    int n_layers = net->n_in_layers + net->n_ou_layers + net->n_de_layers;

    for (int i=1; i<n_layers; i++) {
        int n_units = net->layers[i]->n_units;
        int n_units_prev = net->layers[i-1]->n_units;
        net->layers[i]->weights = matrix_new(n_units_prev, n_units);
        net->layers[i]->bias = matrix_new(net->batch_size, n_units);
        Matrix* weights = net->layers[i]->weights;
        Matrix* bias = net->layers[i]->bias;

        matrix_fill(bias, 0.0);
        switch (net->activation->type)
        {
        case SIGMOID:
            matrix_fill_normal_distribution(weights, 0.0, 2.0/(n_units + n_units_prev));
            break;

        case RELU:
        case LRELU:
        case ELU:
            matrix_fill_normal_distribution(weights, 0.0, 2.0/n_units_prev);
            break;

        default:
            printf("Unknown activation type.");
            break;
        }
    }

    net->compiled = true;
}

void neural_net_link_layers(NeuralNet* net) {
    int n_layers = net->n_in_layers + net->n_ou_layers + net->n_de_layers;
    for (int i=0; i<n_layers; i++) {
        if (i > 0) {
            net->layers[i]->prev_layer = net->layers[i-1];
        }
        if (i < n_layers - 1) {
            net->layers[i]->next_layer = net->layers[i+1];
        }
    }
}

void neural_net_info(NeuralNet* net) {
    if (net->compiled) {
        printf("lp  layer   n_units   output_shape\n");
        printf("------------------------------------\n");
        int n_layers = net->n_in_layers + net->n_ou_layers + net->n_de_layers;
        for (int i=0; i<n_layers; i++) {
            int n_units = net->layers[i]->n_units;
            int batch_size = net->batch_size;
            char* layer = NULL;
            switch (net->layers[i]->l_type)
            {
            case INPUT:
                layer = "Input";
                break;
            case OUTPUT:
                layer = "Output";
                break;
            case DEEP:
                layer = "Deep";
                break;
            default:
                layer = "Undefined";
                break;
            }
            printf("%d  %s  %d  (%d x %d)\n", i, layer, n_units, batch_size, n_units);
            printf("------------------------------------\n");
        }
    }
    else {
        printf("Model has not been compiled. Please compile before running \"neural_net_info\".");
    }
}

Layer* layer_new(LayerType l_type, int n_units, NeuralNet* net) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->l_type = l_type;
    layer->n_units = n_units;
    layer->activation = NULL;
    layer->weights = NULL;
    layer->bias = NULL;
    layer->prev_layer = NULL;
    layer->next_layer = NULL;
    layer->net_backref = net;

    return layer;
}

void layer_free(Layer* layer) {
    if (layer->l_type != INPUT && layer->net_backref->compiled) {
        matrix_free(layer->activation);
        layer->activation = NULL;
        matrix_free(layer->weights);
        layer->weights = NULL;
        matrix_free(layer->bias);
        layer->bias = NULL;
    }
    
    free(layer);
}

void add_input_layer(int n_units, NeuralNet* net) {
    Layer* input_l = layer_new(INPUT, n_units, net);
    int n_layers = net->n_in_layers + net->n_ou_layers + net->n_de_layers;
    net->layers[n_layers] = input_l;
    net->n_in_layers++;
}

void add_output_layer(int n_units, NeuralNet* net) {
    Layer* output_l = layer_new(OUTPUT, n_units, net);
    int n_layers = net->n_in_layers + net->n_ou_layers + net->n_de_layers;
    net->layers[n_layers] = output_l;
    net->n_ou_layers++;
}

void add_deep_layer(int n_units, NeuralNet* net) {
    Layer* deep_l = layer_new(DEEP, n_units, net);
    int n_layers = net->n_in_layers + net->n_ou_layers + net->n_de_layers;
    net->layers[n_layers] = deep_l;
    net->n_de_layers++;
}