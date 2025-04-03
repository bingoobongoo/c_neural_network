#include "network.h"
#include <time.h>
#include <sys/time.h>

NeuralNet* neural_net_new(ActivationType activation_type, CostType cost_type, double activation_param, int batch_size, double learning_rate) {
    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    net->n_layers = 0;
    net->activation = activation_new(activation_type, activation_param);
    net->cost = cost_new(cost_type);
    net->optimizer = optimizer_new(SGD, learning_rate);
    net->batch_score = score_new(batch_size);
    net->train_batch = NULL;
    net->label_batch = NULL;
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
    score_free(net->batch_score);
    batch_free(net->train_batch);
    batch_free(net->label_batch);
    free(net);
}

void neural_net_compile(NeuralNet* net) {
    // 1. link layers
    // 2. initialize weight and bias matrices
    // 3. initialize static storage for gradients, activation and z
    // 4. initialize static storage for auxiliary gradients
    // 5. initialize batches
    neural_net_link_layers(net);

    for (int i=1; i<net->n_layers; i++) {
        Layer* layer = net->layers[i];
        int n_units = layer->n_units;
        int n_units_prev = net->layers[i-1]->n_units;
        layer->weight = matrix_new(n_units_prev, n_units);
        layer->bias = matrix_new(net->batch_size, n_units);
        Matrix* weight = layer->weight;
        Matrix* bias = layer->bias;
        
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

        layer->activation = matrix_new(net->batch_size, n_units);
        layer->z = matrix_new(net->batch_size, n_units);
        layer->delta = matrix_new(net->batch_size, n_units);
        layer->weight_gradient = matrix_new(n_units_prev, n_units);
        layer->bias_gradient = matrix_new(net->batch_size, n_units);

        layer->dCost_dA = matrix_new(net->batch_size, n_units);
        layer->dActivation_dZ = matrix_new(net->batch_size, n_units);
        layer->dZ_dW_t = matrix_new(n_units_prev, net->batch_size);
        if (layer->l_type != OUTPUT)
            layer->dZnext_dA_t = matrix_new(layer->next_layer->n_units, n_units); 
        layer->dCost_dZ_col_sum = matrix_new(1, n_units);
    }

    net->train_batch = batch_new(net->batch_size, net->layers[0]->n_units);
    net->label_batch = batch_new(net->batch_size, net->layers[net->n_layers-1]->n_units);

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
        printf("Activation function: %s\n", net->activation->name);
        printf("Cost function: %s\n", net->cost->name);
        printf("Optimizer: %s\n", net->optimizer->name);
        printf("------------------------------------\n\n");
    }
    else {
        printf("Model has not been compiled. Please compile before running \"neural_net_info\".");
    }
}

void fit(Matrix* x_train, Matrix* y_train, int n_epochs, double validation, NeuralNet* net) {
    for (int epoch=0; epoch<n_epochs; epoch++) {
        int start_idx = 0;
        int i = 0;
        int training_thresh = (1.0 - validation) * x_train->n_rows;
        int n_batches = ceil(x_train->n_rows / (double)net->batch_size);
        double* avg_errs = (double*)malloc(n_batches * sizeof(double));

        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (start_idx; start_idx<training_thresh - net->batch_size; start_idx+=net->batch_size, i++) {
            batchify_into(x_train, start_idx, net->train_batch);
            batchify_into(y_train, start_idx, net->label_batch);
            forward_prop(net);
            avg_errs[i] = get_batch_error(net);
            back_prop(net);
        }
        gettimeofday(&end, NULL);
        double epoch_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

        double  sum=0.0;
        for (int j=0; j<i; j++) {
            sum += avg_errs[j];
        }
        double avg_epoch_error = sum / (double)i;

        int val_batches = n_batches - i;
        double* acc = (double*)malloc(val_batches * sizeof(double));
        i = 0;

        for (start_idx; start_idx<x_train->n_rows - net->batch_size; start_idx+=net->batch_size, i++) {
            batchify_into(x_train, start_idx, net->train_batch);
            batchify_into(y_train, start_idx, net->label_batch);
            forward_prop(net);
            Matrix* y_pred = net->layers[net->n_layers-1]->activation;
            Matrix* y_true = net->label_batch->data;
            score_batch(net->batch_score, y_pred, y_true);
            acc[i] = net->batch_score->accuracy;
        }

        sum = 0.0;
        for (int j=0; j<i; j++) {
            sum += acc[j];
        }
        double avg_epoch_val_acc = sum / (double)i;

        shuffle_data_inplace(x_train, y_train);

        printf("Epoch: %d   err: %f   val_acc: %.3f   time: %.3fs\n", epoch, avg_epoch_error, avg_epoch_val_acc, epoch_time);

        free(avg_errs); free(acc);
    }
}

void forward_prop(NeuralNet* net) {
    Matrix* input = net->train_batch->data;
    for (int i=0; i<net->n_layers; i++) {
        Layer* layer = net->layers[i];

        switch (layer->l_type)
        {
        case INPUT:
            layer->activation = input;
            break;

        case DEEP:
        case OUTPUT:
            matrix_dot_into(input, layer->weight, layer->z);
            matrix_add_into(layer->z, layer->bias, layer->z);
            apply_activation_func_into(net->activation, layer->z, layer->activation);
            input = layer->activation;
            break;
        }
    }
}

// TODO: move calculations of derivatives to cost.h and activation.h
void back_prop(NeuralNet* net) {
    Matrix* label_m = net->label_batch->data;

    for (int i=net->n_layers-1; i>=0; i--) {
        Layer* layer = net->layers[i];
        switch (layer->l_type)
        {
        case OUTPUT: {
            // delta gradient (dCost_dZ) calculations:
            apply_cost_dA_into(net->cost, layer->activation, label_m, layer->dCost_dA);
            apply_activation_dZ_into(net->activation, layer->z, layer->dActivation_dZ);
            matrix_multiply_into(layer->dCost_dA, layer->dActivation_dZ, layer->delta);

            // weight gradient (dCost_dW) calculations:
            matrix_transpose_into(layer->prev_layer->activation, layer->dZ_dW_t);
            matrix_dot_into(layer->dZ_dW_t, layer->delta, layer->weight_gradient);

            // bias gradient (dCost_dB) calculations:
            matrix_sum_axis_into(layer->delta, 1, layer->dCost_dZ_col_sum);
            matrix_multiplicate_into(layer->dCost_dZ_col_sum, 1, net->label_batch->batch_size, layer->bias_gradient);

            break;
        }
        
        case DEEP: {
            // delta gradient (dCost_dZ) calculations:
            matrix_transpose_into(layer->next_layer->weight, layer->dZnext_dA_t); 
            matrix_dot_into(layer->next_layer->delta, layer->dZnext_dA_t, layer->dCost_dA);
            apply_activation_dZ_into(net->activation, layer->z, layer->dActivation_dZ);
            matrix_multiply_into(layer->dCost_dA, layer->dActivation_dZ, layer->delta);

            // weight gradient (dCost_dW) calculations:
            matrix_transpose_into(layer->prev_layer->activation, layer->dZ_dW_t);
            matrix_dot_into(layer->dZ_dW_t, layer->delta, layer->weight_gradient);

            // bias gradient (dCost_dB) calculations:
            matrix_sum_axis_into(layer->delta, 1, layer->dCost_dZ_col_sum);
            matrix_multiplicate_into(layer->dCost_dZ_col_sum, 1, net->label_batch->batch_size, layer->bias_gradient);

            break;
        }

        case INPUT: {
            // here we just update all the weights and biases
            for (int j=1; j<net->n_layers; j++) {
                Layer* current_layer = net->layers[j];

                update_params_inplace(current_layer->weight, current_layer->weight_gradient, net->optimizer);
                update_params_inplace(current_layer->bias, current_layer->bias_gradient, net->optimizer);
            }
            break;
        }
        }
    }
}

double get_batch_error(NeuralNet* net) {
    Matrix* err_m = apply_cost_func(net->cost, net->layers[net->n_layers-1]->activation, net->label_batch->data);
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
    layer->dCost_dA = NULL;
    layer->dActivation_dZ = NULL;
    layer->dZ_dW_t = NULL;
    layer->dZnext_dA_t = NULL;
    layer->dCost_dZ_col_sum = NULL;
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
        if (layer->dCost_dA != NULL) {
            matrix_free(layer->dCost_dA);
            layer->dCost_dA = NULL;
        }
        if (layer->dActivation_dZ != NULL) {
            matrix_free(layer->dActivation_dZ);
            layer->dActivation_dZ = NULL;
        }
        if (layer->dZ_dW_t != NULL) {
            matrix_free(layer->dZ_dW_t);
            layer->dZ_dW_t = NULL;
        }
        if (layer->dZnext_dA_t != NULL) {
            matrix_free(layer->dZnext_dA_t);
            layer->dZnext_dA_t = NULL;
        }
        if (layer->dCost_dZ_col_sum != NULL) {
            matrix_free(layer->dCost_dZ_col_sum);
            layer->dCost_dZ_col_sum = NULL;
        }
    }
    
    free(layer);
}

void add_input_layer(int n_units, NeuralNet* net) {
    Layer* input_l = layer_new(INPUT, n_units, net);
    net->layers[net->n_layers] = input_l;
    net->n_layers++;
}

void add_output_layer(int n_units, NeuralNet* net) {
    Layer* output_l = layer_new(OUTPUT, n_units, net);
    net->layers[net->n_layers] = output_l;
    net->n_layers++;
}

void add_deep_layer(int n_units, NeuralNet* net) {
    Layer* deep_l = layer_new(DEEP, n_units, net);
    net->layers[net->n_layers] = deep_l;
    net->n_layers++;
}