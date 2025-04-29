#include "network.h"

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
    net->is_cnn = false;

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
        
        // init actiavtion and cost
        switch (layer->l_type)
        {
        case OUTPUT:
            if (layer_get_n_units(layer) == 1)
                layer->activation = activation_new(SIGMOID, 0.0);
            if (layer_get_n_units(layer) > 1)
                layer->activation = activation_new(SOFTMAX, 0.0);
            net->cost->loss_m = matrix_new(net->batch_size, layer_get_n_units(layer));
            break;
        
        default:
            layer->activation = activation_new(net->activation->type, net->activation->activation_param);
            break;
        }

        // init weight and bias
        switch (layer->l_type)
        {
        case DEEP:
        case OUTPUT: {
            layer->cache.dense.weight = matrix_new(
                layer_get_n_units(layer->prev_layer),
                layer_get_n_units(layer)
            );
            layer->cache.dense.bias = matrix_new(
                net->batch_size,
                layer_get_n_units(layer)
            );
            break;
        }

        case CONV_2D:
            layer->cache.conv.filter = tensor4D_new(
                layer->params.conv.filter_size,
                layer->params.conv.filter_size,
                layer->prev_layer->cache.conv.output->n_channels,
                layer->params.conv.n_filters
            );
            layer->cache.conv.bias = tensor4D_new(
                layer->params.conv.filter_size,
                layer->params.conv.filter_size,
                1,
                layer->params.conv.n_filters
            );
            break;
        }

        // init cache
        switch (layer->l_type)
        {
        case DEEP:
        case OUTPUT: {
            layer->cache.dense.output = matrix_new(
                net->batch_size, 
                layer_get_n_units(layer)
            );
            layer->cache.dense.z = matrix_new(
                net->batch_size,
                layer_get_n_units(layer)
            );
            layer->cache.dense.delta = matrix_new(
                net->batch_size,
                layer_get_n_units(layer)
            );
            layer->cache.dense.weight_gradient = matrix_new(
                layer_get_n_units(layer->prev_layer),
                layer_get_n_units(layer)
            );
            layer->cache.dense.bias_gradient = matrix_new(
                net->batch_size,
                layer_get_n_units(layer)
            );
            layer->cache.dense.dCost_dA = matrix_new(
                net->batch_size,
                layer_get_n_units(layer)
            );
            layer->cache.dense.dActivation_dZ = matrix_new(
                net->batch_size,
                layer_get_n_units(layer)
            );
            layer->cache.dense.dZ_dW_t = matrix_new(
                layer_get_n_units(layer->prev_layer),
                net->batch_size
            );
            if (layer->l_type == DEEP) {
                layer->cache.dense.dZnext_dA_t = matrix_new(
                    layer_get_n_units(layer->next_layer),
                    layer_get_n_units(layer)
                );
            }
            layer->cache.dense.dCost_dZ_col_sum = matrix_new(
                1,
                layer_get_n_units(layer)
            );

            Matrix* weight = layer->cache.dense.weight;
            Matrix* bias = layer->cache.dense.bias;
            matrix_fill(bias, 0.0);

            switch (net->activation->type)
            {
            case SIGMOID:
                matrix_fill_normal_distribution(
                    weight, 
                    0.0, 
                    2.0/(layer_get_n_units(layer) + layer_get_n_units(layer->prev_layer))
                );
                break;

            case RELU:
            case LRELU:
            case ELU:
                matrix_fill_normal_distribution(
                    weight, 
                    0.0, 
                    2.0/layer_get_n_units(layer->prev_layer)
                );
                break;

            default:
                printf("Unknown activation type.");
                break;
            }
            break;
        }

        case CONV_2D: {
            int input_height = layer->prev_layer->cache.conv.output->n_rows;
            int input_width = layer->prev_layer->cache.conv.output->n_cols;
            int filter_height = layer->params.conv.filter_size;
            int filter_width = layer->params.conv.filter_size;
            int stride = layer->params.conv.stride;

            layer->cache.conv.output = tensor4D_new(
                floor((input_height - filter_height) / stride) + 1,
                floor((input_width - filter_width) / stride) + 1,
                layer->params.conv.n_filters,
                net->batch_size
            );
            layer->cache.conv.z = tensor4D_new(
                floor((input_height - filter_height) / stride) + 1,
                floor((input_width - filter_width) / stride) + 1,
                layer->params.conv.n_filters,
                net->batch_size
            );
            layer->cache.conv.delta = tensor4D_new(
                floor((input_height - filter_height) / stride) + 1,
                floor((input_width - filter_width) / stride) + 1,
                layer->params.conv.n_filters,
                net->batch_size
            );
            layer->cache.conv.filter_gradient = tensor4D_new(
                layer->params.conv.filter_size,
                layer->params.conv.filter_size,
                layer->prev_layer->cache.conv.output->n_channels,
                layer->params.conv.n_filters
            );
            layer->cache.conv.bias_gradient = tensor4D_new(
                layer->params.conv.filter_size,
                layer->params.conv.filter_size,
                1,
                layer->params.conv.n_filters
            );
            layer->cache.conv.rotated_filter = tensor4D_new(
                layer->params.conv.filter_size,
                layer->params.conv.filter_size,
                layer->prev_layer->cache.conv.output->n_channels,
                layer->params.conv.n_filters
            );
            layer->cache.conv.dCost_dA = tensor4D_new(
                floor((input_height - filter_height) / stride) + 1,
                floor((input_width - filter_width) / stride) + 1,
                layer->params.conv.n_filters,
                net->batch_size
            );
            layer->cache.conv.dActivation_dZ = tensor4D_new(
                floor((input_height - filter_height) / stride) + 1,
                floor((input_width - filter_width) / stride) + 1,
                layer->params.conv.n_filters,
                net->batch_size
            );

            layer->params.conv.n_units = 
                layer->cache.conv.output->n_rows *
                layer->cache.conv.output->n_cols *
                layer->params.conv.n_filters;

            Tensor4D* filter = layer->cache.conv.filter;
            Tensor4D* bias = layer->cache.conv.bias;

            for (int i=0; i<bias->n_filters; i++) {
                Tensor3D* t3d = bias->filters[i];
                for (int j=0; j<t3d->n_channels; j++) {
                    matrix_fill(t3d->channels[j], 0.0);
                }
            }

            switch (net->activation->type)
            {
            case SIGMOID:
                for (int i=0; i<filter->n_filters; i++) {
                    Tensor3D* t3d = filter->filters[i];
                    for (int j=0; j<t3d->n_channels; j++) {
                        matrix_fill_normal_distribution(
                            t3d->channels[j],
                            0.0,
                            2.0/(layer_get_n_units(layer) + layer_get_n_units(layer->prev_layer))
                        );
                    }
                }
                break;
            
            case RELU:
            case LRELU:
            case ELU:
                for (int i=0; i<filter->n_filters; i++) {
                    Tensor3D* t3d = filter->filters[i];
                    for (int j=0; j<t3d->n_channels; j++) {
                        matrix_fill_normal_distribution(
                            t3d->channels[j],
                            0.0,
                            2.0/layer_get_n_units(layer->prev_layer)
                        );
                    }
                }
                break;
            
            default:
                printf("Unknown activation type.");
                break;
            }
        }
        break;

        }
    }

    // optimizer compilation
    // for conv layers, need to change optimizer cache to store
    // tensor3D instead of matrix 
    switch (net->optimizer->type)
    {
    case MOMENTUM:
    case NESTEROV:
        MomentumConfig* mom = (MomentumConfig*)net->optimizer->settings;
        mom->n_layers = net->n_layers;

        mom->weight_momentum = (Matrix**)malloc(mom->n_layers * sizeof(Matrix*));
        mom->bias_momentum = (Matrix**)malloc(mom->n_layers * sizeof(Matrix*));

        mom->weight_momentum[0] = NULL;
        mom->bias_momentum[0] = NULL;

        for (int i=1; i<mom->n_layers; i++) {
            Layer* layer = net->layers[i];

            mom->weight_momentum[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );
            matrix_fill(mom->weight_momentum[i], 0.0);

            mom->bias_momentum[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            matrix_fill(mom->bias_momentum[i], 0.0);
        }
        break;
    
    case ADAGRAD:
        AdaGradConfig* ada = (AdaGradConfig*)net->optimizer->settings;
        ada->n_layers = net->n_layers;
        ada->weight_s = (Matrix**)malloc(ada->n_layers * sizeof(Matrix*));
        ada->bias_s = (Matrix**)malloc(ada->n_layers * sizeof(Matrix*));
        ada->intermediate_w = (Matrix**)malloc(ada->n_layers * sizeof(Matrix*));
        ada->intermediate_b = (Matrix**)malloc(ada->n_layers * sizeof(Matrix*));

        ada->weight_s[0] = NULL;
        ada->bias_s[0] = NULL;
        ada->intermediate_w[0] = NULL;
        ada->intermediate_b[0] = NULL;

        for (int i=1; i<ada->n_layers; i++) {
            Layer* layer = net->layers[i];

            ada->weight_s[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );
            matrix_fill(ada->weight_s[i], 0.0);

            ada->bias_s[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            matrix_fill(ada->bias_s[i], 0.0);

            ada->intermediate_w[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );
            matrix_fill(ada->intermediate_w[i], 0.0);

            ada->intermediate_b[i] = matrix_new(
                net->batch_size, 
                layer_get_n_units(layer)
            );
            matrix_fill(ada->intermediate_b[i], 0.0);
        }
        break;
    
    case ADAM:
        AdamConfig* adam = (AdamConfig*)net->optimizer->settings;;
        adam->n_layers = net->n_layers;

        adam->weight_m = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->weight_m_corr = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->weight_s = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->weight_s_corr = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->intermediate_w = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->bias_m = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->bias_m_corr = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->bias_s = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->bias_s_corr = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));
        adam->intermediate_b = (Matrix**)malloc(adam->n_layers * sizeof(Matrix*));

        adam->weight_m[0] = NULL;
        adam->weight_m_corr[0] = NULL;
        adam->weight_s[0] = NULL;
        adam->weight_s_corr[0] = NULL;
        adam->intermediate_w[0] = NULL;
        adam->bias_m[0] = NULL;
        adam->bias_m_corr[0] = NULL;
        adam->bias_s[0] = NULL;
        adam->bias_s_corr[0] = NULL;
        adam->intermediate_b[0] = NULL;

        for (int i=1; i<adam->n_layers; i++) {
            Layer* layer = net->layers[i];

            adam->weight_m[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );
            adam->weight_m_corr[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );
            adam->weight_s[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );
            adam->weight_s_corr[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );
            adam->intermediate_w[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );

            matrix_fill(adam->weight_m[i], 0.0);
            matrix_fill(adam->weight_m_corr[i], 0.0);
            matrix_fill(adam->weight_s[i], 0.0);
            matrix_fill(adam->weight_s_corr[i], 0.0);
            matrix_fill(adam->intermediate_w[i], 0.0);
            
            adam->bias_m[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            adam->bias_m_corr[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            adam->bias_s[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            adam->bias_s_corr[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            adam->intermediate_b[i] = matrix_new(net->batch_size, layer_get_n_units(layer));

            matrix_fill(adam->bias_m[i], 0.0);
            matrix_fill(adam->bias_m_corr[i], 0.0);
            matrix_fill(adam->bias_s[i], 0.0);
            matrix_fill(adam->bias_s_corr[i], 0.0);
            matrix_fill(adam->intermediate_b[i], 0.0);
        }
        break;
    }

    net->label_batch = batch_matrix_new(
        net->batch_size, 
        layer_get_n_units(net->layers[net->n_layers-1])
    );

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
            int n_units = layer_get_n_units(layer);
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
                params = n_units * layer_get_n_units(layer->prev_layer) + n_units;
                trainable_params += params;
                break;
            case DEEP:
                l_name = "Deep";
                params = n_units * layer_get_n_units(layer->prev_layer) + n_units;
                trainable_params += params;
                break;
            case CONV_2D:
                l_name = "Conv2D";
                params = layer->params.conv.n_units + layer->params.conv.n_filters;
                trainable_params += params;
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
        //printf("Output activation: %s\n", net->layers[net->n_layers-1]->activation->name);
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

    Tensor4D* x_train_tensor = NULL;
    Tensor4D* x_train_split_tensor = NULL;
    Tensor4D* x_val_split_tensor = NULL;
    Matrix* x_train_split_mat = NULL;
    Matrix* x_val_split_mat = NULL;

    int train_batches;
    int val_batches;

    // change allocation - useless x_train_tensor
    // better to directly create splits from tensor4D new and matrix slice
    if (net->is_cnn) {
        x_train_tensor = matrix_to_tensor4D(
            x_train,
            net->train_batch->data.tensor->n_rows,
            net->train_batch->data.tensor->n_cols,
            net->train_batch->data.tensor->n_channels 
        );
        tensor4D_slice_into(
            x_train_tensor,
            0,
            training_size,
            x_train_split_tensor
        );
        tensor4D_slice_into(
            x_train_tensor,
            training_size,
            val_size,
            x_val_split_tensor
        );
        train_batches = ceil(x_train_split_tensor->n_filters / (double)net->batch_size);
        val_batches = ceil(x_val_split_tensor->n_filters / (double)net->batch_size);
    }
    else {
        x_train_split_mat = matrix_slice_rows(x_train, 0, training_size);
        x_val_split_mat = matrix_slice_rows(x_train, training_size, val_size);
        train_batches = ceil(x_train_split_mat->n_rows / (double)net->batch_size);
        val_batches = ceil(x_val_split_mat->n_rows / (double)net->batch_size); 
    }
    
    Matrix* y_train_split = matrix_slice_rows(y_train, 0, training_size);
    Matrix* y_val_split = matrix_slice_rows(y_train, training_size, val_size);

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
            if (net->is_cnn) {
                batchify_tensor_into(x_train_split_tensor, start_idx, net->train_batch);
            }
            else {
                batchify_matrix_into(x_train_split_mat, start_idx, net->train_batch);
            }
            batchify_matrix_into(y_train_split, start_idx, net->label_batch);
            forward_prop(net, true);
            avg_loss[i] = get_avg_batch_loss(net->cost, net->layers[net->n_layers-1]->cache.dense.output, net->label_batch->data.matrix);
            Matrix* y_pred = net->layers[net->n_layers-1]->cache.dense.output;
            Matrix* y_true = net->label_batch->data.matrix;
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
            if (net->is_cnn) {
                batchify_tensor_into(x_val_split_tensor, start_idx, net->train_batch);
            }
            else {
                batchify_matrix_into(x_val_split_mat, start_idx, net->train_batch);
            }
            batchify_matrix_into(y_val_split, start_idx, net->label_batch);
            forward_prop(net, false);
            Matrix* y_pred = net->layers[net->n_layers-1]->cache.dense.output;
            Matrix* y_true = net->label_batch->data.matrix;
            score_batch(net->batch_score, y_pred, y_true);
            val_acc[i] = net->batch_score->accuracy;
        }

        sum = 0.0;
        for (int j=0; j<i; j++) {
            sum += val_acc[j];
        }
        avg_epoch_val_acc = sum / (double)i;

        if (net->is_cnn) {
            shuffle_tensor4D_inplace(x_train_split_tensor, y_train_split);
        }
        else {
            shuffle_matrix_inplace(x_train_split_mat, y_train_split);
        }

        printf("Epoch: %d/%d   loss: %f   train_acc: %.4f   val_acc: %.4f   time: %.3fs\n", epoch+1, n_epochs, avg_epoch_loss, avg_epoch_train_acc, avg_epoch_val_acc, epoch_time);
    }

    if (net->optimizer->type == ADAM) {
        AdamConfig* adam = (AdamConfig*)net->optimizer->settings;
        adam->ctr = 1;
    }

    if (net->is_cnn) {
        tensor4D_free(x_train_tensor);
        tensor4D_free(x_train_split_tensor);
        tensor4D_free(x_val_split_tensor);
    }
    else {
        matrix_free(x_train_split_mat);
        matrix_free(x_val_split_mat);
    }
    
    free(avg_loss); free(val_acc); free(train_acc);
    matrix_free(y_train_split);
    matrix_free(y_val_split);
}

void score(Matrix* x_test, Matrix* y_test, NeuralNet* net) {
    Tensor4D* x_test_tensor = NULL;
    int n_batches;

    if (net->is_cnn) {
        x_test_tensor = matrix_to_tensor4D(
            x_test,
            net->train_batch->data.tensor->n_rows,
            net->train_batch->data.tensor->n_cols,
            net->train_batch->data.tensor->n_channels 
        );
        n_batches = ceil(x_test_tensor->n_filters / (double)net->batch_size);
    }
    else {
        n_batches = ceil(x_test->n_rows / (double)net->batch_size);
    }

    int start_idx = 0;
    int i = 0;
    double* acc = (double*)malloc(n_batches * sizeof(double));

    for (start_idx; start_idx<x_test->n_rows - net->batch_size; start_idx+=net->batch_size, i++) {
        if (net->is_cnn) {
            batchify_tensor_into(x_test_tensor, start_idx, net->train_batch);
        }
        else {
            batchify_matrix_into(x_test, start_idx, net->train_batch);
        }
        batchify_matrix_into(y_test, start_idx, net->label_batch);
        forward_prop(net, false);
        Matrix* y_pred = net->layers[net->n_layers-1]->cache.dense.output;
        Matrix* y_true = net->label_batch->data.matrix;
        score_batch(net->batch_score, y_pred, y_true);
        acc[i] = net->batch_score->accuracy;
    }

    double sum = 0.0;
    for (int j=0; j<i; j++) {
        sum += acc[j];
    }
    double test_acc = sum / (double)i;

    printf("Test acc: %.3f\n", test_acc);

    if (net->is_cnn) {
        tensor4D_free(x_test_tensor);
    }

    free(acc);
}

void confusion_matrix(Matrix* x_test, Matrix* y_test, NeuralNet* net) {
    Tensor4D* x_test_tensor = NULL;
    if (net->is_cnn) {
        x_test_tensor = matrix_to_tensor4D(
            x_test,
            net->train_batch->data.tensor->n_rows,
            net->train_batch->data.tensor->n_cols,
            net->train_batch->data.tensor->n_channels
        );
    }

    int start_idx = 0;
    Matrix* conf_m = matrix_new(y_test->n_cols, y_test->n_cols);
    matrix_fill(conf_m, 0.0);

    for (start_idx; start_idx<x_test->n_rows - net->batch_size; start_idx += net->batch_size) {
        if (net->is_cnn) {
            batchify_tensor_into(x_test_tensor, start_idx, net->train_batch);
        }
        else {
            batchify_matrix_into(x_test, start_idx, net->train_batch);
        }
        batchify_matrix_into(y_test, start_idx, net->label_batch);
        forward_prop(net, false);
        Matrix* y_pred = net->layers[net->n_layers-1]->cache.dense.output;
        Matrix* y_true = net->label_batch->data.matrix;
        update_confusion_matrix(net->batch_score, y_pred, y_true, conf_m);
    }

    matrix_print(conf_m);

    if (net->is_cnn) {
        tensor4D_free(x_test_tensor);
    }

    matrix_free(conf_m);
}

void forward_prop(NeuralNet* net, bool training) {
    Batch* input = net->train_batch; 
    for (int i=0; i<net->n_layers; i++) {
        Layer* layer = net->layers[i];

        switch (layer->l_type)
        {
        case INPUT:
            if (net->is_cnn) {
                layer->cache.conv.output = input->data.tensor;
            }
            else {
                layer->cache.dense.output = input->data.matrix;
            }
            break;

        case DEEP:
        case OUTPUT:
            if (net->optimizer->type == NESTEROV && training) {
                MomentumConfig* mom = (MomentumConfig*)net->optimizer->settings;
                matrix_add_into(layer->cache.dense.weight, mom->weight_momentum[i], layer->cache.dense.weight);
                matrix_add_into(layer->cache.dense.bias, mom->bias_momentum[i], layer->cache.dense.bias);
            }
            matrix_dot_into(layer->prev_layer->cache.dense.output, layer->cache.dense.weight, layer->cache.dense.z);
            matrix_add_into(layer->cache.dense.z, layer->cache.dense.bias, layer->cache.dense.z);
            layer->activation->y_true_batch = net->label_batch;
            apply_activation_func_into(layer->activation, layer->cache.dense.z, layer->cache.dense.output);
            break;

        case CONV_2D:
            conv2d_forward(
                layer->prev_layer->cache.conv.output,
                layer->cache.conv.filter,
                layer->cache.conv.bias,
                layer->params.conv.stride,
                layer->cache.conv.z
            );

            for (int i=0; i<net->batch_size; i++) {
                for (int j=0; j<layer->params.conv.n_filters; j++) {
                    apply_activation_func_into(
                        layer->activation,
                        layer->cache.conv.z->filters[i]->channels[j],
                        layer->cache.conv.output->filters[i]->channels[j]
                    );
                }
            }
            break;
        }
    }
}

void back_prop(NeuralNet* net) {
    Matrix* label_m = net->label_batch->data.matrix;

    for (int i=net->n_layers-1; i>=0; i--) {
        Layer* layer = net->layers[i];
        switch (layer->l_type)
        {
        case OUTPUT: {
            // delta gradient (dCost_dZ) calculations:
            apply_cost_dA_into(
                net->cost, 
                layer->cache.dense.output, 
                label_m, 
                layer->cache.dense.dCost_dA
            );
            apply_activation_dZ_into(
                layer->activation, 
                layer->cache.dense.z, 
                layer->cache.dense.dActivation_dZ
            );
            matrix_multiply_into(
                layer->cache.dense.dCost_dA, 
                layer->cache.dense.dActivation_dZ, 
                layer->cache.dense.delta
            );

            // weight gradient (dCost_dW) calculations:
            matrix_transpose_into(
                layer->prev_layer->cache.dense.output, 
                layer->cache.dense.dZ_dW_t
            );
            matrix_dot_into(
                layer->cache.dense.dZ_dW_t, 
                layer->cache.dense.delta, 
                layer->cache.dense.weight_gradient
            );

            // bias gradient (dCost_dB) calculations:
            matrix_sum_axis_into(
                layer->cache.dense.delta, 
                1, 
                layer->cache.dense.dCost_dZ_col_sum
            );
            matrix_multiplicate_into(
                layer->cache.dense.dCost_dZ_col_sum, 
                1, 
                net->label_batch->batch_size, 
                layer->cache.dense.bias_gradient
            );

            break;
        }
        
        case DEEP: {
            // delta gradient (dCost_dZ) calculations:
            matrix_transpose_into(
                layer->next_layer->cache.dense.weight, 
                layer->cache.dense.dZnext_dA_t
            ); 
            matrix_dot_into(
                layer->next_layer->cache.dense.delta, 
                layer->cache.dense.dZnext_dA_t, 
                layer->cache.dense.dCost_dA
            );
            apply_activation_dZ_into(
                layer->activation, 
                layer->cache.dense.z, 
                layer->cache.dense.dActivation_dZ
            );
            matrix_multiply_into(
                layer->cache.dense.dCost_dA, 
                layer->cache.dense.dActivation_dZ, 
                layer->cache.dense.delta
            );

            // weight gradient (dCost_dW) calculations:
            matrix_transpose_into(
                layer->prev_layer->cache.dense.output, 
                layer->cache.dense.dZ_dW_t
            );
            matrix_dot_into(
                layer->cache.dense.dZ_dW_t, 
                layer->cache.dense.delta, 
                layer->cache.dense.weight_gradient
            );

            // bias gradient (dCost_dB) calculations:
            matrix_sum_axis_into(
                layer->cache.dense.delta, 
                1, 
                layer->cache.dense.dCost_dZ_col_sum
            );
            matrix_multiplicate_into(
                layer->cache.dense.dCost_dZ_col_sum, 
                1, 
                net->label_batch->batch_size, 
                layer->cache.dense.bias_gradient
            );

            break;
        }

        case CONV_2D: {
            // delta gradient (dCost_dZ) calculation
            // 1. dCost_dA
            Tensor4D* delta = layer->cache.conv.delta;
            Tensor4D* filter = layer->cache.conv.filter;
            Tensor4D* rot_filter = layer->cache.conv.rotated_filter;
            tensor4D_rot180_into(filter, rot_filter);
            Tensor4D* dCost_dA = layer->cache.conv.dCost_dA;
            for (int n=0; n<delta->n_filters; n++) {
                for (int c=0; c<delta->n_channels; c++) {
                    for (int dh=0; dh<delta->n_rows; dh++) {
                        for (int dw=0; dw<delta->n_cols; dw++) {
                            
                            double delta_val = delta
                                ->filters[n]
                                ->channels[c]
                                ->entries[dh][dw];
                            
                            for (int fc=0; fc<filter->n_channels; fc++) {
                                for (int fh=0; fh<filter->n_rows; fh++) {
                                    for (int fw=0; fw<filter->n_cols; fw++) {

                                        int out_h = dh + fh;
                                        int out_w = dw + fw;
                                        dCost_dA
                                        ->filters[n]
                                        ->channels[fc]
                                        ->entries[out_h][out_w]
                                        += delta_val * rot_filter
                                        ->filters[c]
                                        ->channels[fc]
                                        ->entries[fh][fw];

                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 2. dA_dZ
            Tensor4D* z = layer->cache.conv.z;
            Tensor4D* dA_dZ = layer->cache.conv.dActivation_dZ;
            for (int n=0; n<z->n_filters; n++) {
                for (int c=0; c<z->n_channels; c++) {
                    Matrix* z_m = z->filters[n]->channels[c];
                    Matrix* dA_dZ_m = dA_dZ->filters[n]->channels[c];
                    apply_activation_dZ_into(layer->activation, z_m, dA_dZ_m);
                }
            }

            // 3. dCost_dZ
            for (int n=0; n<z->n_filters; n++) {
                for(int c=0; c<z->n_channels; c++) {
                    Matrix* dCost_dA_m = dCost_dA->filters[n]->channels[c];
                    Matrix* dA_dZ_m = dA_dZ->filters[n]->channels[c];
                    Matrix* delta_m = delta->filters[n]->channels[c];
                    matrix_multiply_into(dCost_dA_m, dA_dZ_m, delta_m);
                }
            }

            // weight gradient (dCost_dw) calculation
            Tensor4D* input = layer->prev_layer->cache.conv.output;
            Tensor4D* filter_grad = layer->cache.conv.filter_gradient;
            int stride = layer->params.conv.stride;
            for (int delta_c=0; delta_c<delta->n_channels; delta_c++) {
                for (int input_c=0; input_c<input->n_channels; input_c++) {
                    for (int gh=0; gh<filter_grad->n_rows; gh++) {
                        for (int gw=0; gw<filter_grad->n_cols; gw++) {

                            double grad = 0.0;

                            for (int n=0; n<input->n_filters; n++) {
                                for (int dh=0; dh<delta->n_rows; dh++) {
                                    for (int dw=0; dw<delta->n_cols; dw++) {

                                        int ih = dh*stride + gh;
                                        int iw = dw*stride + gw;

                                        double delta_val = delta
                                        ->filters[n]
                                        ->channels[delta_c]
                                        ->entries[dh][dw];

                                        double input_val = input
                                        ->filters[n]
                                        ->channels[input_c]
                                        ->entries[ih][iw];

                                        grad += delta_val * input_val;
                                    }
                                }
                            }

                            filter_grad
                            ->filters[delta_c]
                            ->channels[input_c]
                            ->entries[gh][gw] = grad;
                        }
                    }
                }
            }

            // bias gradient (dCost_dB) calculation
            Tensor4D* bias_grad = layer->cache.conv.bias_gradient;
            for (int c=0; c<delta->n_channels; c++) {

                double bias_sum=0.0;

                for (int n=0; n<delta->n_filters; n++) {
                    for (int dh=0; dh<delta->n_rows; dh++) {
                        for (int dw=0; dw<delta->n_cols; dw++) {
                            bias_sum += delta
                            ->filters[n]
                            ->channels[c]
                            ->entries[dh][dw];
                        }
                    }
                }
                
                matrix_fill(
                    bias_grad->filters[c]->channels[0],
                    bias_sum
                );
            }
            break;
        }

        case INPUT: {
            // here we just update all the weights and biases
            for (int j=1; j<net->n_layers; j++) {
                Layer* layer = net->layers[j];

                if (net->optimizer->type == NESTEROV) {
                    MomentumConfig* mom = (MomentumConfig*)net->optimizer->settings;
                    matrix_subtract_into(
                        layer->cache.dense.weight, 
                        mom->weight_momentum[j], 
                        layer->cache.dense.weight
                    );
                    matrix_subtract_into(
                        layer->cache.dense.bias, 
                        mom->bias_momentum[j], 
                        layer->cache.dense.bias
                    );
                }

                switch (layer->l_type)
                {
                case DEEP:
                case OUTPUT:
                    net->optimizer->update_weights(
                        layer->cache.dense.weight, 
                        layer->cache.dense.weight_gradient, 
                        net->optimizer, 
                        j
                    );
                    net->optimizer->update_bias(
                        layer->cache.dense.bias, 
                        layer->cache.dense.bias_gradient, 
                        net->optimizer, 
                        j
                    );
                    break;

                case CONV_2D:
                    Tensor4D* filter = layer->cache.conv.filter;
                    Tensor4D* filter_grad = layer->cache.conv.filter_gradient;
                    Tensor4D* bias = layer->cache.conv.bias;
                    Tensor4D* bias_grad = layer->cache.conv.bias_gradient;
                    for (int i=0; i<filter->n_filters; i++) {
                        Matrix* bias_m = bias->filters[i]->channels[0];
                        Matrix* bias_grad_m = bias_grad->filters[i]->channels[0];
                        for (int j=0; j<filter->n_channels; j++) {
                            Matrix* filter_m = filter->filters[i]->channels[j];
                            Matrix* filter_grad_m = filter->filters[i]->channels[j];
                            net->optimizer->update_weights(
                                filter_m,
                                filter_grad_m,
                                net->optimizer,
                                j // unsafe! add logic for more complex opt
                            );
                        }
                        net->optimizer->update_bias(
                            bias_m,
                            bias_grad_m,
                            net->optimizer,
                            j // here also
                        );
                    }
                }
            }

            if (net->optimizer->type == ADAM) {
                AdamConfig* adam = (AdamConfig*)net->optimizer->settings;
                adam->ctr++;
            }

            break;
        }
        }
    }
}

Layer* layer_new(LayerType l_type, NeuralNet* net) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->l_type = l_type;
    layer->activation = NULL;
    layer->prev_layer = NULL;
    layer->next_layer = NULL;
    layer->net_backref = net;

    return layer;
}

void layer_free(Layer* layer) { 
    if (layer->net_backref->compiled) {
        if (layer->l_type == INPUT) {
            free(layer);
            return;
        }

        if (layer->l_type == CONV_2D) {
            if (layer->cache.conv.output != NULL) {
                tensor4D_free(layer->cache.conv.output);
                layer->cache.conv.output = NULL;
            }
            if (layer->cache.conv.z != NULL) {
                tensor4D_free(layer->cache.conv.z);
                layer->cache.conv.z = NULL;
            }
            if (layer->cache.conv.filter != NULL) {
                tensor4D_free(layer->cache.conv.filter);
                layer->cache.conv.filter = NULL;
            }
            if (layer->cache.conv.bias != NULL) {
                tensor4D_free(layer->cache.conv.bias);
                layer->cache.conv.bias = NULL;
            }
            if (layer->cache.conv.delta != NULL) {
                tensor4D_free(layer->cache.conv.delta);
                layer->cache.conv.delta = NULL;
            }
            if (layer->cache.conv.filter_gradient != NULL) {
                tensor4D_free(layer->cache.conv.filter_gradient);
                layer->cache.conv.filter_gradient = NULL;
            }
            if (layer->cache.conv.rotated_filter != NULL) {
                tensor4D_free(layer->cache.conv.rotated_filter);
                layer->cache.conv.rotated_filter = NULL;
            }
            if (layer->cache.conv.bias_gradient != NULL) {
                tensor4D_free(layer->cache.conv.bias_gradient);
                layer->cache.conv.bias_gradient = NULL;
            }
            if (layer->cache.conv.dCost_dA != NULL) {
                tensor4D_free(layer->cache.conv.dCost_dA);
                layer->cache.conv.dCost_dA = NULL;
            }
            if (layer->cache.conv.dActivation_dZ != NULL) {
                tensor4D_free(layer->cache.conv.dActivation_dZ);
                layer->cache.conv.dActivation_dZ = NULL;
            }
        }

        else {
            if (layer->cache.dense.output != NULL) {
                matrix_free(layer->cache.dense.output);
                layer->cache.dense.output = NULL;
            }
            if (layer->cache.dense.z != NULL) {
                matrix_free(layer->cache.dense.z);
                layer->cache.dense.z = NULL;
            }
            if (layer->cache.dense.weight != NULL) {
                matrix_free(layer->cache.dense.weight);
                layer->cache.dense.weight = NULL;
            }
            if (layer->cache.dense.bias != NULL) {
                matrix_free(layer->cache.dense.bias);
                layer->cache.dense.bias = NULL;
            }
            if (layer->cache.dense.delta != NULL) {
                matrix_free(layer->cache.dense.delta);
                layer->cache.dense.delta = NULL;
            }
            if (layer->cache.dense.weight_gradient != NULL) {
                matrix_free(layer->cache.dense.weight_gradient);
                layer->cache.dense.weight_gradient = NULL;
            }
            if (layer->cache.dense.bias_gradient != NULL) {
                matrix_free(layer->cache.dense.bias_gradient);
                layer->cache.dense.bias_gradient = NULL;
            }
            if (layer->cache.dense.dCost_dA != NULL) {
                matrix_free(layer->cache.dense.dCost_dA);
                layer->cache.dense.dCost_dA = NULL;
            }
            if (layer->cache.dense.dActivation_dZ != NULL) {
                matrix_free(layer->cache.dense.dActivation_dZ);
                layer->cache.dense.dActivation_dZ = NULL;
            }
            if (layer->cache.dense.dZ_dW_t != NULL) {
                matrix_free(layer->cache.dense.dZ_dW_t);
                layer->cache.dense.dZ_dW_t = NULL;
            }
            if (layer->cache.dense.dZnext_dA_t != NULL) {
                matrix_free(layer->cache.dense.dZnext_dA_t);
                layer->cache.dense.dZnext_dA_t = NULL;
            }
            if (layer->cache.dense.dCost_dZ_col_sum != NULL) {
                matrix_free(layer->cache.dense.dCost_dZ_col_sum);
                layer->cache.dense.dCost_dZ_col_sum = NULL;
            }
        }
        if (layer->activation != NULL) {
            free(layer->activation);
            layer->activation = NULL;
        }
    }
    free(layer);
}

int layer_get_n_units(Layer* layer) {
    switch (layer->l_type)
    {
    case INPUT:
    case DEEP:
    case OUTPUT:
        return layer->params.dense.n_units;
        break;
    
    case CONV_2D:
        return layer->params.conv.n_units;
        break;
    }

    exit(1);
}

void add_input_layer(int n_units, NeuralNet* net) {
    Layer* input_l = layer_new(INPUT, net);
    input_l->params.dense.n_units = n_units;

    net->train_batch = batch_matrix_new(
        net->batch_size,
        n_units
    );

    input_l->cache.dense.output = net->train_batch->data.matrix;

    net->layers[net->n_layers] = input_l;
    net->n_layers++;
}

void add_conv_input_layer(int n_rows, int n_cols, int n_channels, NeuralNet* net) {
    Layer* input_l = layer_new(INPUT, net);
    input_l->params.conv.n_filters = net->batch_size;
    input_l->params.conv.filter_size = 0;
    input_l->params.conv.stride = 0;
    input_l->params.conv.n_units = n_rows * n_cols * n_channels;

    net->train_batch = batch_tensor_new(
        net->batch_size,
        n_rows,
        n_cols,
        n_channels
    );

    input_l->cache.conv.output = net->train_batch->data.tensor;

    net->is_cnn = true;
    net->layers[net->n_layers] = input_l;
    net->n_layers++;
}

void add_output_layer(int n_units, NeuralNet* net) {
    Layer* output_l = layer_new(OUTPUT, net);
    output_l->params.dense.n_units = n_units;
    net->layers[net->n_layers] = output_l;
    net->n_layers++;
}

void add_deep_layer(int n_units, NeuralNet* net) {
    Layer* deep_l = layer_new(DEEP, net);
    deep_l->params.dense.n_units = n_units;
    net->layers[net->n_layers] = deep_l;
    net->n_layers++;
}

void add_conv_layer(int n_filters, int filter_size, int stride, NeuralNet* net) {
    Layer* conv_l = layer_new(CONV_2D, net);
    conv_l->params.conv.n_filters = n_filters;
    conv_l->params.conv.filter_size = filter_size;
    conv_l->params.conv.stride = stride;
    net->is_cnn = true;
    net->layers[net->n_layers] = conv_l;
    net->n_layers++;
}