#include "network.h"

NeuralNet* neural_net_new(ActivationType activation_type, CostType cost_type, double activation_param, int batch_size, double learning_rate) {
    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    net->n_layers = 0;
    net->n_in_layers = 0;
    net->n_ou_layers = 0;
    net->n_de_layers = 0;
    net->activation = activation_new(activation_type, activation_param);
    net->cost = cost_new(cost_type);
    net->optimizer = optimizer_new(SGD, learning_rate);
    net->batch_size = batch_size;
    net->layers = (Layer**)malloc(100 * sizeof(Layer*));
    net->compiled = false;

    return net;
}

void neural_net_free(NeuralNet* net) {
    for (int i=0; i<net->n_layers; i++) {
        layer_free(net->layers[i]);
    }
    
    free(net->layers);
    free(net->activation);
    free(net->cost);
    free(net->optimizer);
    free(net);
}

void neural_net_compile(NeuralNet* net) {
    // 1. link layers
    // 2. initialize weight and bias matrices
    neural_net_link_layers(net);

    for (int i=1; i<net->n_layers; i++) {
        int n_units = net->layers[i]->n_units;
        int n_units_prev = net->layers[i-1]->n_units;
        net->layers[i]->weight = matrix_new(n_units_prev, n_units);
        net->layers[i]->bias = matrix_new(net->batch_size, n_units);
        Matrix* weight = net->layers[i]->weight;
        Matrix* bias = net->layers[i]->bias;
        
        matrix_fill(bias, 0.0);
        switch (net->activation->type)
        {
        case SIGMOID:
            matrix_fill_normal_distribution(weight, 0.0, 2.0/(n_units + n_units_prev));
            break;

        case RELU:
        case LRELU:
        case ELU:
            matrix_fill_normal_distribution(weight, 0.0, 2.0/n_units_prev);
            break;

        default:
            printf("Unknown activation type.");
            break;
        }
    }

    net->compiled = true;
}

void neural_net_link_layers(NeuralNet* net) {
    for (int i=0; i<net->n_layers; i++) {
        if (i > 0) {
            net->layers[i]->prev_layer = net->layers[i-1];
        }
        if (i < net->n_layers - 1) {
            net->layers[i]->next_layer = net->layers[i+1];
        }
    }
}

void neural_net_info(NeuralNet* net) {
    if (net->compiled) {
        printf("lp  layer   n_units   output_shape\n");
        printf("------------------------------------\n");
        for (int i=0; i<net->n_layers; i++) {
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

void fit(Matrix* x_train, Matrix* y_train, int n_epochs, NeuralNet* net) {
    for (int epoch=0; epoch<n_epochs; epoch++) {
        int start_idx = 0;
        int i = 0;
        int n_batches = ceil(x_train->n_rows / (double)net->batch_size);
        double* avg_errs = (double*)malloc(n_batches * sizeof(double));

        for (start_idx; start_idx<x_train->n_rows - net->batch_size; start_idx+=net->batch_size, i++) {
            Batch train_batch = batchify(x_train, start_idx, net->batch_size, true);
            Batch label_batch = batchify(y_train, start_idx, net->batch_size, true);
            forward_prop(&train_batch, net);
            avg_errs[i] = get_batch_error(&label_batch, net);
            back_prop(&label_batch, net);
            batch_free(&train_batch);
            batch_free(&label_batch);
        }

        shuffle_data_inplace(x_train, y_train);
        double  sum=0.0;
        for (int j=0; j<i; j++) {
            sum += avg_errs[j];
        }
        double avg_epoch_error = sum / (double)i;

        printf("Epoch: %d   Err: %f\n", epoch, avg_epoch_error);
        free(avg_errs);
    }
}

void forward_prop(Batch* batch, NeuralNet* net) {
    Matrix* input = batch->data;
    for (int i=0; i<net->n_layers; i++) {
        Layer* layer = net->layers[i];

        switch (layer->l_type)
        {
        case INPUT:
            layer->activation = input;
            break;

        case DEEP:
        case OUTPUT:
            Matrix* prod = matrix_dot(input, layer->weight);
            layer->z = matrix_add(prod, layer->bias);
            layer->activation = apply_activation_func(net->activation, layer->z);
            input = layer->activation;
            matrix_free(prod);
            break;
        }
    }
}

// TODO: move calculations of derivatives to cost.h and activation.h
void back_prop(Batch* label_batch, NeuralNet* net) {
    Matrix* label_m = label_batch->data;

    for (int i=net->n_layers-1; i>=0; i--) {
        Layer* layer = net->layers[i];
        switch (layer->l_type)
        {
        case OUTPUT: {
            // delta gradient (dCost_dZ) calculations:
            Matrix* dCost_dA = apply_cost_dA(net->cost, layer->activation, label_m);
            Matrix* dActivation_dZ = apply_activation_dZ(net->activation, layer->z);
            Matrix* dCost_dZ = matrix_multiply(dCost_dA, dActivation_dZ);
            matrix_free(dCost_dA); matrix_free(dActivation_dZ);

            // weight gradient (dCost_dW) calculations:
            Matrix* dZ_dW_t = matrix_transpose(layer->prev_layer->activation);
            Matrix* dCost_dW = matrix_dot(dZ_dW_t, dCost_dZ);
            matrix_free(dZ_dW_t);

            // bias gradient (dCost_dB) calculations:
            Matrix* dCost_dZ_col_sum = matrix_sum_axis(dCost_dZ, 1);
            Matrix* dCost_dB = matrix_multiplicate(dCost_dZ_col_sum, 1, label_batch->batch_size);
            matrix_free(dCost_dZ_col_sum);

            layer->delta = dCost_dZ;
            layer->weight_gradient = dCost_dW;
            layer->bias_gradient = dCost_dB;

            break;
        }
        
        case DEEP: {
            // delta gradient (dCost_dZ) calculations:
            Matrix* dZnext_dA_t = matrix_transpose(layer->next_layer->weight); 
            Matrix* dCost_dA = matrix_dot(layer->next_layer->delta, dZnext_dA_t);
            Matrix* dActivation_dZ = apply_activation_dZ(net->activation, layer->z);
            Matrix* dCost_dZ = matrix_multiply(dCost_dA, dActivation_dZ);
            matrix_free(dZnext_dA_t); matrix_free(dCost_dA); matrix_free(dActivation_dZ);

            // weight gradient (dCost_dW) calculations:
            Matrix* dZ_dW_t = matrix_transpose(layer->prev_layer->activation);
            Matrix* dCost_dW = matrix_dot(dZ_dW_t, dCost_dZ);
            matrix_free(dZ_dW_t);

            // bias gradient (dCost_dB) calculations:
            Matrix* dCost_dZ_col_sum = matrix_sum_axis(dCost_dZ, 1);
            Matrix* dCost_dB = matrix_multiplicate(dCost_dZ_col_sum, 1, label_batch->batch_size);
            matrix_free(dCost_dZ_col_sum);

            layer->delta = dCost_dZ;
            layer->weight_gradient = dCost_dW;
            layer->bias_gradient = dCost_dB;
            break;
        }
        
        case INPUT: {
            // here we just update all the weights and biases
            for (int j=1; j<net->n_layers; j++) {
                Layer* current_layer = net->layers[j];

                Matrix* new_weight = update_params(current_layer->weight, current_layer->weight_gradient, net->optimizer);
                Matrix* new_bias = update_params(current_layer->bias, current_layer->bias_gradient, net->optimizer);

                matrix_assign(&current_layer->weight, new_weight);
                matrix_assign(&current_layer->bias, new_bias);

                // also free memory of all gradients, z and activation matrices
                matrix_free(current_layer->delta);
                current_layer->delta = NULL;
                matrix_free(current_layer->weight_gradient);
                current_layer->weight_gradient = NULL;
                matrix_free(current_layer->bias_gradient);
                current_layer->bias_gradient = NULL;
                matrix_free(current_layer->z);
                current_layer->z = NULL;
                matrix_free(current_layer->activation);
                current_layer->activation = NULL;
            }
            break;
        }
        }
    }
}

double get_batch_error(Batch* label_batch, NeuralNet* net) {
    Matrix* err_m = apply_cost_func(net->cost, net->layers[net->n_layers-1]->activation, label_batch->data);
    double err_avg = matrix_average(err_m);
    matrix_free(err_m);

    return err_avg;
}

Layer* layer_new(LayerType l_type, int n_units, NeuralNet* net) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->l_type = l_type;
    layer->n_units = n_units;
    layer->activation = NULL;
    layer->z = NULL;
    layer->weight = NULL;
    layer->bias = NULL;
    layer->delta = NULL;
    layer->weight_gradient = NULL;
    layer->bias_gradient = NULL;
    layer->prev_layer = NULL;
    layer->next_layer = NULL;
    layer->net_backref = net;

    return layer;
}

void layer_free(Layer* layer) { 
    if (layer->l_type != INPUT && layer->net_backref->compiled) {
        if (layer->activation != NULL) {
            matrix_free(layer->activation);
            layer->activation = NULL;
        }
        if (layer->z != NULL) {
            matrix_free(layer->z);
            layer->z = NULL;
        }
        if (layer->weight != NULL) {
            matrix_free(layer->weight);
            layer->weight = NULL;
        }
        if (layer->bias != NULL) {
            matrix_free(layer->bias);
            layer->bias = NULL;
        }
        if (layer->delta != NULL) {
            matrix_free(layer->delta);
            layer->delta = NULL;
        }
        if (layer->weight_gradient != NULL) {
            matrix_free(layer->weight_gradient);
            layer->weight_gradient = NULL;
        }
        if (layer->bias_gradient != NULL) {
            matrix_free(layer->bias_gradient);
            layer->bias_gradient = NULL;
        }
    }
    
    free(layer);
}

void add_input_layer(int n_units, NeuralNet* net) {
    Layer* input_l = layer_new(INPUT, n_units, net);
    net->layers[net->n_layers] = input_l;
    net->n_in_layers++;
    net->n_layers++;
}

void add_output_layer(int n_units, NeuralNet* net) {
    Layer* output_l = layer_new(OUTPUT, n_units, net);
    net->layers[net->n_layers] = output_l;
    net->n_ou_layers++;
    net->n_layers++;
}

void add_deep_layer(int n_units, NeuralNet* net) {
    Layer* deep_l = layer_new(DEEP, n_units, net);
    net->layers[net->n_layers] = deep_l;
    net->n_de_layers++;
    net->n_layers++;
}