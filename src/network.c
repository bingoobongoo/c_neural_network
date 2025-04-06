#include "network.h"
#include <time.h>
#include <sys/time.h>

NeuralNet* neural_net_new(Optimizer* opt, ActivationType act_type, double act_param, CostType cost_type, int batch_size) {
    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    net->n_layers = 0;
    net->activation = activation_new(act_type, act_param);
    net->cost = cost_new(cost_type);
    net->optimizer = opt;
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
    cost_free(net->cost);
    net->optimizer->optimizer_free(net->optimizer);
    score_free(net->batch_score);
    batch_free(net->train_batch);
    batch_free(net->label_batch);
    free(net);
}

void neural_net_compile(NeuralNet* net) {
    // 1. link layers
    // 2. initialize activation for each layer
    // 3. initialize weight and bias matrices
    // 4. initialize static storage for gradients, output and z
    // 5. initialize static storage for auxiliary gradients
    // 6. initialize static storage for optimizer matrices
    // 7. initialize batches
    neural_net_link_layers(net);

    for (int i=1; i<net->n_layers; i++) {
        Layer* layer = net->layers[i];
        int n_units = layer->n_units;
        int n_units_prev = net->layers[i-1]->n_units;
        
        switch (layer->l_type)
        {
        case OUTPUT:
            if (n_units == 1)
                layer->activation = activation_new(SIGMOID, 0.0);
            if (n_units > 1)
                layer->activation = activation_new(SOFTMAX, 0.0);
            net->cost->loss_m = matrix_new(net->batch_size, layer->n_units);
            break;
        
        default:
            layer->activation = activation_new(net->activation->type, net->activation->activation_param);
            break;
        }

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

        // static gradients, output and z storage
        layer->output = matrix_new(net->batch_size, n_units);
        layer->z = matrix_new(net->batch_size, n_units);
        layer->delta = matrix_new(net->batch_size, n_units);
        layer->weight_gradient = matrix_new(n_units_prev, n_units);
        layer->bias_gradient = matrix_new(net->batch_size, n_units);

        // static auxiliary gradients storage
        layer->dCost_dA = matrix_new(net->batch_size, n_units);
        layer->dActivation_dZ = matrix_new(net->batch_size, n_units);
        layer->dZ_dW_t = matrix_new(n_units_prev, net->batch_size);
        if (layer->l_type != OUTPUT)
            layer->dZnext_dA_t = matrix_new(layer->next_layer->n_units, n_units); 
        layer->dCost_dZ_col_sum = matrix_new(1, n_units);
    }

    // optimizer compilation
    switch (net->optimizer->type)
    {
    case MOMENTUM:
        MomentumConfig* mom = (MomentumConfig*)net->optimizer->settings;
        mom->n_layers = net->n_layers;
        mom->weight_momentum = (Matrix**)malloc(mom->n_layers * sizeof(Matrix*));
        mom->bias_momentum = (Matrix**)malloc(mom->n_layers * sizeof(Matrix*));
        mom->weight_momentum[0] = NULL;
        mom->bias_momentum[0] = NULL;
        for (int i=1; i<mom->n_layers; i++) {
            Layer* layer = net->layers[i];

            mom->weight_momentum[i] = matrix_new(layer->prev_layer->n_units, layer->n_units);
            matrix_fill(mom->weight_momentum[i], 0.0);

            mom->bias_momentum[i] = matrix_new(net->batch_size, layer->n_units);
            matrix_fill(mom->bias_momentum[i], 0.0);
        }
        break;
    
    default:
        break;
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
        long long trainable_params = 0;
        long long non_trainable_params = 0;

        printf("lp  layer   n_units   output_shape  params\n");
        printf("------------------------------------\n");
        for (int i=0; i<net->n_layers; i++) {
            Layer* layer = net->layers[i];
            int n_units = layer->n_units;
            int batch_size = net->batch_size;
            long long params = 0;
            char* l_name = NULL;
            switch (layer->l_type)
            {
            case INPUT:
                l_name = "Input";
                params = n_units;
                non_trainable_params += params;
                break;
            case OUTPUT:
                l_name = "Output";
                params = n_units * layer->prev_layer->n_units + n_units;
                trainable_params += params;
                break;
            case DEEP:
                l_name = "Deep";
                params = n_units * layer->prev_layer->n_units + n_units;
                trainable_params += params;
                break;
            default:
                l_name = "Undefined";
                break;
            }
            printf("%d  %s  %d  (%d x %d)  %lld\n", i, l_name, n_units, batch_size, n_units, params);
            printf("------------------------------------\n");
        }
        printf("Trainable params: %lld\n", trainable_params);
        printf("Non-trainable params: %lld\n", non_trainable_params);
        printf("Activation function: %s\n", net->activation->name);
        printf("Output activation: %s\n", net->layers[net->n_layers-1]->activation->name);
        printf("Cost function: %s\n", net->cost->name);
        printf("Optimizer: %s\n", net->optimizer->name);
        printf("------------------------------------\n\n");
    }
    else {
        printf("Model has not been compiled. Please compile before running \"neural_net_info\".");
    }
}

void fit(Matrix* x_train, Matrix* y_train, int n_epochs, double validation, NeuralNet* net) {
    int training_size = (1.0 - validation) * x_train->n_rows;
    int val_size = validation * x_train->n_rows;
    Matrix* x_train_split = matrix_slice_rows(x_train, 0, training_size);
    Matrix* y_train_split = matrix_slice_rows(y_train, 0, training_size);
    Matrix* x_val_split = matrix_slice_rows(x_train, training_size, val_size);
    Matrix* y_val_split = matrix_slice_rows(y_train, training_size, val_size);

    int train_batches = ceil(x_train_split->n_rows / (double)net->batch_size);
    int val_batches = ceil(x_val_split->n_rows / (double)net->batch_size);
    double* avg_loss = (double*)malloc(train_batches * sizeof(double));
    double* train_acc = (double*)malloc(train_batches * sizeof(double));
    double* val_acc = (double*)malloc(val_batches * sizeof(double));

    int start_idx = 0;
    int i = 0;

    double epoch_time;
    double sum;
    double avg_epoch_loss;
    double avg_epoch_train_acc;
    double avg_epoch_val_acc;

    for (int epoch=0; epoch<n_epochs; epoch++) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (start_idx; start_idx<training_size - net->batch_size; start_idx+=net->batch_size, i++) {
            batchify_into(x_train_split, start_idx, net->train_batch);
            batchify_into(y_train_split, start_idx, net->label_batch);
            forward_prop(net);
            avg_loss[i] = get_avg_batch_loss(net->cost, net->layers[net->n_layers-1]->output, net->label_batch->data);
            Matrix* y_pred = net->layers[net->n_layers-1]->output;
            Matrix* y_true = net->label_batch->data;
            score_batch(net->batch_score, y_pred, y_true);
            train_acc[i] = net->batch_score->accuracy;
            back_prop(net);
        }
        gettimeofday(&end, NULL);
        epoch_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

        sum=0.0;
        for (int j=0; j<i; j++) {
            sum += avg_loss[j];
        }
        avg_epoch_loss = sum / (double)i;

        sum=0.0;
        for (int j=0; j<i; j++) {
            sum += train_acc[j];
        }
        avg_epoch_train_acc = sum / (double)i;

        i = 0;
        for (start_idx=0; start_idx<val_size - net->batch_size; start_idx+=net->batch_size, i++) {
            batchify_into(x_val_split, start_idx, net->train_batch);
            batchify_into(y_val_split, start_idx, net->label_batch);
            forward_prop(net);
            Matrix* y_pred = net->layers[net->n_layers-1]->output;
            Matrix* y_true = net->label_batch->data;
            score_batch(net->batch_score, y_pred, y_true);
            val_acc[i] = net->batch_score->accuracy;
        }

        sum = 0.0;
        for (int j=0; j<i; j++) {
            sum += val_acc[j];
        }
        avg_epoch_val_acc = sum / (double)i;

        shuffle_data_inplace(x_train_split, y_train_split);
        printf("Epoch: %d   loss: %f   train_acc: %.4f   val_acc: %.4f   time: %.3fs\n", epoch, avg_epoch_loss, avg_epoch_train_acc, avg_epoch_val_acc, epoch_time);
    }
    
    free(avg_loss); free(val_acc); free(train_acc);
    matrix_free(x_train_split); matrix_free(y_train_split);
    matrix_free(x_val_split); matrix_free(y_val_split);
}

void score(Matrix* x_test, Matrix* y_test, NeuralNet* net) {
    int start_idx = 0;
    int i = 0;
    int n_batches = ceil(x_test->n_rows / (double)net->batch_size);
    double* acc = (double*)malloc(n_batches * sizeof(double));

    for (start_idx; start_idx<x_test->n_rows - net->batch_size; start_idx+=net->batch_size, i++) {
        batchify_into(x_test, start_idx, net->train_batch);
        batchify_into(y_test, start_idx, net->label_batch);
        forward_prop(net);
        Matrix* y_pred = net->layers[net->n_layers-1]->output;
        Matrix* y_true = net->label_batch->data;
        score_batch(net->batch_score, y_pred, y_true);
        acc[i] = net->batch_score->accuracy;
    }

    double sum = 0.0;
    for (int j=0; j<i; j++) {
        sum += acc[j];
    }
    double test_acc = sum / (double)i;

    printf("Test acc: %.3f\n", test_acc);

    free(acc);
}

void confusion_matrix(Matrix* x_test, Matrix* y_test, NeuralNet* net) {
    int start_idx = 0;

    Matrix* conf_m = matrix_new(y_test->n_cols, y_test->n_cols);
    matrix_fill(conf_m, 0.0);

    for (start_idx; start_idx<x_test->n_rows - net->batch_size; start_idx += net->batch_size) {
        batchify_into(x_test, start_idx, net->train_batch);
        batchify_into(y_test, start_idx, net->label_batch);
        forward_prop(net);
        Matrix* y_pred = net->layers[net->n_layers-1]->output;
        Matrix* y_true = net->label_batch->data;
        update_confusion_matrix(net->batch_score, y_pred, y_true, conf_m);
    }

    matrix_print(conf_m);
    matrix_free(conf_m);
}

void forward_prop(NeuralNet* net) {
    Matrix* input = net->train_batch->data;
    for (int i=0; i<net->n_layers; i++) {
        Layer* layer = net->layers[i];

        switch (layer->l_type)
        {
        case INPUT:
            layer->output = input;
            break;

        case DEEP:
            matrix_dot_into(input, layer->weight, layer->z);
            matrix_add_into(layer->z, layer->bias, layer->z);
            apply_activation_func_into(layer->activation, layer->z, layer->output);
            input = layer->output;
            break;
        case OUTPUT:
            matrix_dot_into(input, layer->weight, layer->z);
            matrix_add_into(layer->z, layer->bias, layer->z);
            layer->activation->y_true_batch = net->label_batch;
            apply_activation_func_into(layer->activation, layer->z, layer->output);
            input = layer->output;
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
            apply_cost_dA_into(net->cost, layer->output, label_m, layer->dCost_dA);
            apply_activation_dZ_into(layer->activation, layer->z, layer->dActivation_dZ);
            matrix_multiply_into(layer->dCost_dA, layer->dActivation_dZ, layer->delta);

            // weight gradient (dCost_dW) calculations:
            matrix_transpose_into(layer->prev_layer->output, layer->dZ_dW_t);
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
            apply_activation_dZ_into(layer->activation, layer->z, layer->dActivation_dZ);
            matrix_multiply_into(layer->dCost_dA, layer->dActivation_dZ, layer->delta);

            // weight gradient (dCost_dW) calculations:
            matrix_transpose_into(layer->prev_layer->output, layer->dZ_dW_t);
            matrix_dot_into(layer->dZ_dW_t, layer->delta, layer->weight_gradient);

            // bias gradient (dCost_dB) calculations:
            matrix_sum_axis_into(layer->delta, 1, layer->dCost_dZ_col_sum);
            matrix_multiplicate_into(layer->dCost_dZ_col_sum, 1, net->label_batch->batch_size, layer->bias_gradient);

            break;
        }

        case INPUT: {
            // here we just update all the weights and biases
            for (int j=1; j<net->n_layers; j++) {
                Layer* layer = net->layers[j];

                net->optimizer->update_weights(layer->weight, layer->weight_gradient, net->optimizer, j);
                net->optimizer->update_bias(layer->bias, layer->bias_gradient, net->optimizer, j);
            }
            break;
        }
        }
    }
}

Layer* layer_new(LayerType l_type, int n_units, NeuralNet* net) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->l_type = l_type;
    layer->n_units = n_units;
    layer->activation = NULL;
    layer->output = NULL;
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
            free(layer->activation);
            layer->activation = NULL;
        }
        if (layer->output != NULL) {
            matrix_free(layer->output);
            layer->output = NULL;
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