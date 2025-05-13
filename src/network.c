#include "network.h"

NeuralNet* neural_net_new(Optimizer* opt, ActivationType act_type, float act_param, CostType cost_type, int batch_size) {
    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    net->n_layers = 0;
    net->activation = activation_new(act_type, act_param);
    net->cost = cost_new(cost_type);
    net->optimizer = opt;
    net->batch_score = score_new(batch_size);
    net->train_batch = NULL;
    net->label_batch = NULL;
    net->batch_size = batch_size;
    net->layers = (Layer**)malloc(20 * sizeof(Layer*));
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
        switch (layer->l_type)
        {
        case DEEP:
            layer_deep_compile(layer, net);
            break;

        case OUTPUT: 
            layer_output_compile(layer, net);
            break;

        case CONV_2D: 
            layer_conv2D_compile(layer, net);
            break;

        case FLATTEN: 
            layer_flatten_compile(layer, net);
            break;

        case MAX_POOL: 
            layer_max_pool_compile(layer, net);
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
            case INPUT: {
                l_name = "InputDense";
                params = n_units;
                non_trainable_params += params;
                printf("%d  %s  %d  (%d x %d)  %lld\n", i, l_name, n_units, batch_size, n_units, params);
                printf("------------------------------------\n");
                break;
            }
            case CONV_2D_INPUT: {
                l_name = "InputConv2D";
                int out_chan = layer->cache.conv.output->n_channels;
                int out_rows = layer->cache.conv.output->n_rows;
                int out_cols = layer->cache.conv.output->n_cols;
                params = n_units;
                non_trainable_params += params;
                printf("%d  %s  %d  (%d x %d x %d x %d)  %lld\n", i, l_name, n_units, batch_size, out_chan, out_rows, out_cols, params);
                printf("------------------------------------\n");
                break;
            }
            case OUTPUT: {
                l_name = "Output";
                params = n_units * layer_get_n_units(layer->prev_layer) + n_units;
                trainable_params += params;
                printf("%d  %s  %d  (%d x %d)  %lld\n", i, l_name, n_units, batch_size, n_units, params);
                printf("------------------------------------\n");
                break;
            }
            case DEEP: {
                l_name = "Deep";
                params = n_units * layer_get_n_units(layer->prev_layer) + n_units;
                trainable_params += params;
                printf("%d  %s  %d  (%d x %d)  %lld\n", i, l_name, n_units, batch_size, n_units, params);
                printf("------------------------------------\n");
                break;
            }
            case CONV_2D: {
                l_name = "Conv2D";
                int out_chan = layer->cache.conv.output->n_channels;
                int out_rows = layer->cache.conv.output->n_rows;
                int out_cols = layer->cache.conv.output->n_cols;
                int filter_size = layer->params.conv.filter_size;
                int filter_chan = layer->cache.conv.filter->n_channels;
                int n_filters = layer->params.conv.n_filters;
                params = n_filters * (pow(filter_size, 2) * filter_chan + 1);
                trainable_params += params;
                printf("%d  %s  %d  (%d x %d x %d x %d)  %lld\n", i, l_name, n_units, batch_size, out_chan, out_rows, out_cols, params);
                printf("------------------------------------\n");
                break;
            }
            case FLATTEN: {
                l_name = "Flatten";
                params = 0;
                printf("%d  %s  %d  (%d x %d)  %lld\n", i, l_name, n_units, batch_size, n_units, params);
                printf("------------------------------------\n");
                break;
            }
            case MAX_POOL: {
                int out_chan = layer->cache.conv.output->n_channels;
                int out_rows = layer->cache.conv.output->n_rows;
                int out_cols = layer->cache.conv.output->n_cols;
                l_name = "MaxPool";
                params = 0;
                printf("%d  %s  %d  (%d x %d x %d x %d)  %lld\n", i, l_name, n_units, batch_size, out_chan, out_rows, out_cols, params);
                printf("------------------------------------\n");
                break;
            }
            default:
                l_name = "Undefined";
                break;
            }
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

void fit(Matrix* x_train, Matrix* y_train, int n_epochs, float validation, NeuralNet* net) {
    int training_size = (1.0 - validation) * x_train->n_rows;
    int val_size = validation * x_train->n_rows;

    Matrix* x_train_split_mat = matrix_slice_rows(x_train, 0, training_size);
    Matrix* x_val_split_mat = matrix_slice_rows(x_train, training_size, val_size);
    Tensor4D* x_train_split_tensor = NULL;
    Tensor4D* x_val_split_tensor = NULL;

    int train_batches;
    int val_batches;

    if (net->is_cnn) {
        x_train_split_tensor = matrix_to_tensor4D(
            x_train_split_mat,
            net->train_batch->data.tensor->n_rows,
            net->train_batch->data.tensor->n_cols,
            net->train_batch->data.tensor->n_channels
        );

        x_val_split_tensor = matrix_to_tensor4D(
            x_val_split_mat,
            net->train_batch->data.tensor->n_rows,
            net->train_batch->data.tensor->n_cols,
            net->train_batch->data.tensor->n_channels
        );

        train_batches = ceil(x_train_split_tensor->n_filters / (float)net->batch_size);
        val_batches = ceil(x_val_split_tensor->n_filters / (float)net->batch_size);
        
        matrix_free(x_train_split_mat);
        matrix_free(x_val_split_mat);
    }
    else {
        train_batches = ceil(x_train_split_mat->n_rows / (float)net->batch_size);
        val_batches = ceil(x_val_split_mat->n_rows / (float)net->batch_size); 
    }
    
    Matrix* y_train_split = matrix_slice_rows(y_train, 0, training_size);
    Matrix* y_val_split = matrix_slice_rows(y_train, training_size, val_size);

    float* avg_loss = (float*)malloc(train_batches * sizeof(float));
    float* train_acc = (float*)malloc(train_batches * sizeof(float));
    float* val_acc = (float*)malloc(val_batches * sizeof(float));

    int start_idx = 0;
    int i = 0;

    float epoch_time;
    float sum;
    float avg_epoch_loss;
    float avg_epoch_train_acc;
    float avg_epoch_val_acc;

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
        epoch_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;;

        sum=0.0;
        for (int j=0; j<i; j++) {
            sum += avg_loss[j];
        }
        avg_epoch_loss = sum / (float)i;

        sum=0.0;
        for (int j=0; j<i; j++) {
            sum += train_acc[j];
        }
        avg_epoch_train_acc = sum / (float)i;

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
        avg_epoch_val_acc = sum / (float)i;

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
        n_batches = ceil(x_test_tensor->n_filters / (float)net->batch_size);
    }
    else {
        n_batches = ceil(x_test->n_rows / (float)net->batch_size);
    }

    int start_idx = 0;
    int i = 0;
    float* acc = (float*)malloc(n_batches * sizeof(float));

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

    float sum = 0.0;
    for (int j=0; j<i; j++) {
        sum += acc[j];
    }
    float test_acc = sum / (float)i;

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

    Matrix* row_sum = matrix_sum_axis(conf_m, 0);
    for (int i=0; i<row_sum->n_cols; i++) {
        for (int j=0; j<row_sum->n_cols; j++) {
            matrix_assign(
                conf_m,
                i,
                j,
                matrix_get(conf_m, i, j) / matrix_get(row_sum, 0, i)
            );
        }
    }

    matrix_print(conf_m);
    
    if (net->is_cnn) {
        tensor4D_free(x_test_tensor);
    }
    
    matrix_free(row_sum);
    matrix_free(conf_m);
}

void forward_prop(NeuralNet* net, bool training) {
    Batch* input = net->train_batch; 
    for (int i=0; i<net->n_layers; i++) {
        Layer* l = net->layers[i];

        switch (l->l_type)
        {
        case INPUT:
            layer_input_fp(l, net);
            break;
        
        case CONV_2D_INPUT:
            layer_conv2D_input_fp(l, net);
            break;

        case DEEP:
        case OUTPUT:
            if (net->optimizer->type == NESTEROV && training) {
                MomentumConfig* mom = (MomentumConfig*)net->optimizer->settings;
                matrix_add_into(l->cache.dense.weight, mom->weight_momentum[i], l->cache.dense.weight);
                matrix_add_into(l->cache.dense.bias, mom->bias_momentum[i], l->cache.dense.bias);
            }

            layer_deep_fp(l, net);
            break;

        case CONV_2D:
            layer_conv2D_fp(l, net);
            break;
            
        case FLATTEN:
            layer_flatten_fp(l, net);
            break;
        
        case MAX_POOL:
            layer_max_pool_fp(l, net);
            break;
        }
    }
}

void back_prop(NeuralNet* net) {
    Matrix* label_m = net->label_batch->data.matrix;
    for (int i=net->n_layers-1; i>=0; i--) {
        Layer* l = net->layers[i];
        switch (l->l_type)
        {
        case OUTPUT: 
            layer_output_bp(l, net);
            break;
        
        case DEEP:
            layer_deep_bp(l, net);
            break;

        case CONV_2D:
            layer_conv2D_bp(l, net);
            break;

        case FLATTEN:
            layer_flatten_bp(l, net);
            break;

        case MAX_POOL: 
            layer_max_pool_bp(l, net);
            break;

        case INPUT: 
        case CONV_2D_INPUT: {
            for (int j=1; j<net->n_layers; j++) {
                l = net->layers[j];

                if (net->optimizer->type == NESTEROV) {
                    MomentumConfig* mom = (MomentumConfig*)net->optimizer->settings;
                    matrix_subtract_into(
                        l->cache.dense.weight, 
                        mom->weight_momentum[j], 
                        l->cache.dense.weight
                    );
                    matrix_subtract_into(
                        l->cache.dense.bias, 
                        mom->bias_momentum[j], 
                        l->cache.dense.bias
                    );
                }

                switch (l->l_type)
                {
                case DEEP:
                case OUTPUT:
                    layer_deep_update_weights(l, net);
                    break;

                case CONV_2D:
                    layer_conv2D_update_weights(l, net);
                    break;
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
        switch (layer->l_type)
        {
        case INPUT:
        case CONV_2D_INPUT:
            free(layer);
            return;
            break;
        
        case CONV_2D:
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
            if (layer->cache.conv.input_im2col != NULL) {
                matrix_free(layer->cache.conv.input_im2col);
                layer->cache.conv.input_im2col = NULL;
            }
            if (layer->cache.conv.kernel_im2col != NULL) {
                matrix_free(layer->cache.conv.kernel_im2col);
                layer->cache.conv.kernel_im2col = NULL;
            }
            if (layer->cache.conv.im2col_dot != NULL) {
                matrix_free(layer->cache.conv.im2col_dot);
                layer->cache.conv.im2col_dot = NULL;
            }
           break;
           
        case DEEP:
        case OUTPUT:
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
            break;

        case FLATTEN:
            if (layer->cache.flat.output != NULL) {
                matrix_free(layer->cache.flat.output);
                layer->cache.flat.output = NULL;
            }
            if (layer->cache.flat.dCost_dA_matrix != NULL) {
                matrix_free(layer->cache.flat.dCost_dA_matrix);
                layer->cache.flat.dCost_dA_matrix = NULL;
            }
            if (layer->cache.flat.dZnext_dA_t != NULL) {
                matrix_free(layer->cache.flat.dZnext_dA_t);
                layer->cache.flat.dZnext_dA_t = NULL;
            }
            break;

        case MAX_POOL: 
            if (layer->cache.conv.output != NULL) {
                tensor4D_free(layer->cache.conv.output);
                layer->cache.conv.output = NULL;
            }
            if (layer->cache.conv.delta != NULL) {
                tensor4D_free(layer->cache.conv.delta);
                layer->cache.conv.delta = NULL;
            }
            break;
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
    
    case CONV_2D_INPUT:
    case CONV_2D:
    case MAX_POOL:
        return layer->params.conv.n_units;
        break;
    
    case FLATTEN:
        return layer->params.flat.n_units;
        break;
    }

    exit(1);
}

Matrix* layer_get_output_matrix(Layer* layer) {
    switch (layer->l_type)
    {
    case INPUT:
    case DEEP:
    case OUTPUT:
        return layer->cache.dense.output;
        break;

    case FLATTEN:
        return layer->cache.flat.output;    
        break;

    default:
        printf("Expected to return matrix, not tensor.");
        exit(1);
    }
}

Tensor4D* layer_get_output_tensor4D(Layer* layer) {
    switch (layer->l_type)
    {
    case CONV_2D_INPUT:
    case CONV_2D:
    case MAX_POOL:
        return layer->cache.conv.output;
        break;
    
    default:
        printf("Expected to return tensor, not matrix.");
        exit(1);
    }
}

Matrix* layer_get_delta_matrix(Layer* layer) {
    switch (layer->l_type)
    {
    case DEEP:
    case OUTPUT:
        return layer->cache.dense.delta;
        break;
    
    case FLATTEN:
        return layer->cache.flat.dCost_dA_matrix;
        break;
    
    default:
        printf("Object doesn't have delta matrix.");
        exit(1);
    }
}
Tensor4D* layer_get_delta_tensor4D(Layer* layer) {
    switch (layer->l_type)
    {
    case CONV_2D:
    case MAX_POOL:
        return layer->cache.conv.delta;
        break;
    
    default:
        printf("Object doesn't have delta tensor.");
        exit(1);
    }
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
    input_l->layer_idx = net->n_layers;
    net->n_layers++;
}

void add_conv_input_layer(int n_rows, int n_cols, int n_channels, NeuralNet* net) {
    Layer* input_l = layer_new(CONV_2D_INPUT, net);
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
    input_l->layer_idx = net->n_layers;
    net->n_layers++;
}

void add_output_layer(int n_units, NeuralNet* net) {
    Layer* output_l = layer_new(OUTPUT, net);
    output_l->params.dense.n_units = n_units;

    net->label_batch = batch_matrix_new(
        net->batch_size, 
        n_units
    );

    net->layers[net->n_layers] = output_l;
    output_l->layer_idx = net->n_layers;
    net->n_layers++;
}

void add_deep_layer(int n_units, NeuralNet* net) {
    Layer* deep_l = layer_new(DEEP, net);
    deep_l->params.dense.n_units = n_units;
    net->layers[net->n_layers] = deep_l;
    deep_l->layer_idx = net->n_layers;
    net->n_layers++;
}

void add_conv_layer(int n_filters, int filter_size, int stride, NeuralNet* net) {
    Layer* conv_l = layer_new(CONV_2D, net);
    conv_l->params.conv.n_filters = n_filters;
    conv_l->params.conv.filter_size = filter_size;
    conv_l->params.conv.stride = stride;
    conv_l->params.conv.n_units = -1;
    net->layers[net->n_layers] = conv_l;
    conv_l->layer_idx = net->n_layers;
    net->n_layers++;
}

void add_flatten_layer(NeuralNet* net) {
    Layer* flat_l = layer_new(FLATTEN, net);
    net->layers[net->n_layers] = flat_l;
    flat_l->layer_idx = net->n_layers;
    net->n_layers++;
}

void add_max_pool_layer(int filter_size, int stride, NeuralNet* net) {
    Layer* max_pool_l = layer_new(MAX_POOL, net);
    max_pool_l->params.conv.n_filters = 1;
    max_pool_l->params.conv.filter_size = filter_size;
    max_pool_l->params.conv.stride = stride;
    max_pool_l->params.conv.n_units = -1;
    net->layers[net->n_layers] = max_pool_l;
    max_pool_l->layer_idx = net->n_layers;
    net->n_layers++;
}

void layer_deep_compile(Layer* l, NeuralNet* net) {
    l->activation = activation_new(
        net->activation->type,
        net->activation->activation_param
    );

    l->cache.dense.weight = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.output = matrix_new(
        net->batch_size, 
        layer_get_n_units(l)
    );
    l->cache.dense.z = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.delta = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.weight_gradient = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias_gradient = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dCost_dA = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dActivation_dZ = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dZ_dW_t = matrix_new(
        layer_get_n_units(l->prev_layer),
        net->batch_size
    );
    l->cache.dense.dZnext_dA_t = matrix_new(
        layer_get_n_units(l->next_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.dCost_dZ_col_sum = matrix_new(
        1,
        layer_get_n_units(l)
    );

    matrix_fill(l->cache.dense.bias, 0.0);
    switch (l->activation->type)
    {
    case SIGMOID:
        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            0.0,
            2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer))
        );
        break;
    
    case RELU:
    case LRELU:
    case ELU:
        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            0.0,
            2.0/layer_get_n_units(l->prev_layer)
        );
        break;
    
    default:
        printf("Unknown activation type.");
        exit(1);
        break;
    }
}

void layer_output_compile(Layer* l, NeuralNet* net) {
    if (layer_get_n_units(l) == 1)
        l->activation = activation_new(SIGMOID, 0.0);
    if (layer_get_n_units(l) > 1)
        l->activation = activation_new(SOFTMAX, 0.0);
    net->cost->loss_m = matrix_new(net->batch_size, layer_get_n_units(l));

    l->cache.dense.weight = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.output = matrix_new(
        net->batch_size, 
        layer_get_n_units(l)
    );
    l->cache.dense.z = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.delta = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.weight_gradient = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias_gradient = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dCost_dA = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dActivation_dZ = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dZ_dW_t = matrix_new(
        layer_get_n_units(l->prev_layer),
        net->batch_size
    );
    l->cache.dense.dCost_dZ_col_sum = matrix_new(
        1,
        layer_get_n_units(l)
    );
    l->cache.dense.dZnext_dA_t = NULL;

    matrix_fill(l->cache.dense.bias, 0.0);
    switch (net->activation->type)
    {
    case SIGMOID:
        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            0.0,
            2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer))
        );
        break;
    
    case RELU:
    case LRELU:
    case ELU:
        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            0.0,
            2.0/layer_get_n_units(l->prev_layer)
        );
        break;
    
    default:
        printf("Unknown activation type.");
        exit(1);
        break;
    }
}

void layer_conv2D_compile(Layer* l, NeuralNet* net) {
    l->activation = activation_new(
        net->activation->type,
        net->activation->activation_param
    );

    int input_height = layer_get_output_tensor4D(l->prev_layer)->n_rows;
    int input_width = layer_get_output_tensor4D(l->prev_layer)->n_cols;
    int input_channels = layer_get_output_tensor4D(l->prev_layer)->n_channels;
    int filter_height = l->params.conv.filter_size;
    int filter_width = l->params.conv.filter_size;
    int stride = l->params.conv.stride;
    int output_height = floor((input_height - filter_height) / stride) + 1;
    int output_width = floor((input_width - filter_width) / stride) + 1;

    l->cache.conv.filter = tensor4D_new(
        l->params.conv.filter_size,
        l->params.conv.filter_size,
        input_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.bias = tensor4D_new(
        output_height,
        output_width,
        1,
        l->params.conv.n_filters
    );
    l->cache.conv.output = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        net->batch_size
    );
    l->cache.conv.z = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        net->batch_size
    );
    l->cache.conv.delta = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        net->batch_size
    );
    l->cache.conv.filter_gradient = tensor4D_new(
        l->params.conv.filter_size,
        l->params.conv.filter_size,
        input_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.bias_gradient = tensor4D_new(
        output_height,
        output_width,
        1,
        l->params.conv.n_filters
    );
    l->cache.conv.dCost_dA = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        net->batch_size
    );
    l->cache.conv.dActivation_dZ = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        net->batch_size
    );
    l->cache.conv.input_im2col = matrix_new(
        output_height * output_width,
        filter_height * filter_width * input_channels
    );
    l->cache.conv.kernel_im2col = matrix_new(
        filter_height * filter_width * input_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.im2col_dot = matrix_new(
        output_height * output_width,
        l->params.conv.n_filters
    );

    l->params.conv.n_units = 
        l->cache.conv.output->n_rows *
        l->cache.conv.output->n_cols *
        l->cache.conv.output->n_channels;
    
    tensor4D_fill(
        l->cache.conv.bias,
        0.0
    );
    switch (l->activation->type)
    {
    case SIGMOID:
        tensor4D_fill_normal_distribution(
            l->cache.conv.filter,
            0.0,
            2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer))
        );
        break;
    
    case RELU:
    case LRELU:
    case ELU:
        tensor4D_fill_normal_distribution(
            l->cache.conv.filter,
            0.0,
            2.0/layer_get_n_units(l->prev_layer)
        );
        break;
    
    default:
        printf("Unknown activation type.");
        exit(1);
        break;
    }
    
    kernel_into_im2col(
        l->cache.conv.filter,
        false,
        l->cache.conv.kernel_im2col
    );
}

void layer_flatten_compile(Layer* l, NeuralNet* net) {
    int prev_chan = layer_get_output_tensor4D(l->prev_layer)->n_channels;
    int prev_rows = layer_get_output_tensor4D(l->prev_layer)->n_rows;
    int prev_cols = layer_get_output_tensor4D(l->prev_layer)->n_cols;
    l->params.flat.n_units = prev_chan * prev_rows * prev_cols;
    l->cache.flat.output = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.flat.dCost_dA_matrix = matrix_new(
        net->batch_size,
        layer_get_n_units(l)
    );
    l->cache.flat.dZnext_dA_t = matrix_new(
        layer_get_n_units(l->next_layer),
        layer_get_n_units(l)
    );
}

void layer_max_pool_compile(Layer* l, NeuralNet* net) {
    int input_height = layer_get_output_tensor4D(l->prev_layer)->n_rows;
    int input_width = layer_get_output_tensor4D(l->prev_layer)->n_cols;
    int input_channels = layer_get_output_tensor4D(l->prev_layer)->n_channels;
    int filter_height = l->params.conv.filter_size;
    int filter_width = l->params.conv.filter_size;
    int stride = l->params.conv.stride;
    int output_height = floor((input_height - filter_height) / stride) + 1;
    int output_width = floor((input_width - filter_width) / stride) + 1;

    l->cache.conv.output = tensor4D_new(
        output_height,
        output_width,
        input_channels,
        net->batch_size
    );
    l->cache.conv.delta = tensor4D_new(
        output_height,
        output_width,
        input_channels,
        net->batch_size
    );
    l->cache.conv.dCost_dA = l->cache.conv.delta;

    l->cache.conv.filter = NULL;
    l->cache.conv.bias = NULL;
    l->cache.conv.z = NULL;
    l->cache.conv.filter_gradient = NULL;
    l->cache.conv.bias_gradient = NULL;
    l->cache.conv.dActivation_dZ = NULL;
    l->cache.conv.input_im2col = NULL;
    l->cache.conv.kernel_im2col = NULL;
    l->cache.conv.im2col_dot = NULL;

    l->params.conv.n_units = 
        l->cache.conv.output->n_rows *
        l->cache.conv.output->n_cols *
        l->cache.conv.output->n_channels;
}

void layer_input_fp(Layer* l, NeuralNet* net) {
    l->cache.dense.output = net->train_batch->data.matrix;
}

void layer_conv2D_input_fp(Layer* l, NeuralNet* net) {
    l->cache.conv.output = net->train_batch->data.tensor;
}

void layer_deep_fp(Layer* l, NeuralNet* net) {
    matrix_dot_into(layer_get_output_matrix(l->prev_layer), l->cache.dense.weight, l->cache.dense.z);
    matrix_add_into(l->cache.dense.z, l->cache.dense.bias, l->cache.dense.z);
    l->activation->y_true_batch = net->label_batch;
    apply_activation_func_into(l->activation, l->cache.dense.z, l->cache.dense.output);
}

void layer_conv2D_fp(Layer* l, NeuralNet* net) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* filter = l->cache.conv.filter;
    Tensor4D* z = l->cache.conv.z;
    Tensor4D* bias = l->cache.conv.bias;
    Tensor4D* output = l->cache.conv.output;
    
    for (int n=0; n<net->batch_size; n++) {
        input_into_im2col(
            input->filters[n],
            filter,
            l->params.conv.stride,
            VALID,
            l->cache.conv.input_im2col 
        );
        im2col_correlate(
            l->cache.conv.input_im2col,
            l->cache.conv.kernel_im2col,
            l->cache.conv.im2col_dot,
            z->filters[n]
        );
        for (int i=0; i<filter->n_filters; i++) {
            matrix_add_into(
                z->filters[n]->channels[i],
                bias->filters[i]->channels[0],
                z->filters[n]->channels[i]
            );
            apply_activation_func_into(
                l->activation,
                z->filters[n]->channels[i],
                output->filters[n]->channels[i]
            );
        }
    }

    // Tensor3D* corr_t3d = tensor3D_new(
    //     z->n_rows,
    //     z->n_cols,
    //     filter->n_channels
    // );
    // for (int n=0; n<net->batch_size; n++) {
    //     for (int i=0; i<filter->n_filters; i++) {
    //         tensor3D_correlate_into(
    //             input->filters[n],
    //             filter->filters[i],
    //             corr_t3d,
    //             l->params.conv.stride,
    //             VALID
    //         );
    //         tensor3D_sum_element_wise_into(
    //             corr_t3d,
    //             z->filters[n]->channels[i]
    //         );
    //         matrix_add_into(
    //             z->filters[n]->channels[i],
    //             bias->filters[i]->channels[0],
    //             z->filters[n]->channels[i]
    //         );
    //         apply_activation_func_into(
    //             l->activation,
    //             z->filters[n]->channels[i],
    //             output->filters[n]->channels[i]
    //         );
    //     }
    // }
    // tensor3D_free(corr_t3d);
}

void layer_flatten_fp(Layer* l, NeuralNet* net) {
    tensor4D_into_matrix(
        layer_get_output_tensor4D(l->prev_layer),
        l->cache.flat.output,
        false,
        false
    );
}

void layer_max_pool_fp(Layer* l, NeuralNet* net) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* output = layer_get_output_tensor4D(l);

    for (int n=0; n<net->batch_size; n++) {
        for (int c=0; c<input->n_channels; c++) {
            matrix_max_pool_into(
                input->filters[n]->channels[c],
                output->filters[n]->channels[c],
                l->params.conv.filter_size,
                l->params.conv.stride
            );
        }
    }
}

void layer_output_bp(Layer*l, NeuralNet* net) {
    // delta gradient (dCost_dZ) calculations:
    apply_cost_dA_into(
        net->cost, 
        l->cache.dense.output, 
        net->label_batch->data.matrix, 
        l->cache.dense.dCost_dA
    );
    apply_activation_dZ_into(
        l->activation, 
        l->cache.dense.z, 
        l->cache.dense.dActivation_dZ
    );
    matrix_multiply_into(
        l->cache.dense.dCost_dA, 
        l->cache.dense.dActivation_dZ, 
        l->cache.dense.delta
    );

    // weight gradient (dCost_dW) calculations:
    matrix_transpose_into(
        layer_get_output_matrix(l->prev_layer), 
        l->cache.dense.dZ_dW_t
    );
    matrix_dot_into(
        l->cache.dense.dZ_dW_t, 
        l->cache.dense.delta, 
        l->cache.dense.weight_gradient
    );

    // bias gradient (dCost_dB) calculations:
    matrix_sum_axis_into(
        l->cache.dense.delta, 
        1, 
        l->cache.dense.dCost_dZ_col_sum
    );
    matrix_multiplicate_into(
        l->cache.dense.dCost_dZ_col_sum, 
        1, 
        net->label_batch->batch_size, 
        l->cache.dense.bias_gradient
    );
}

void layer_deep_bp(Layer* l, NeuralNet* net) {
    // delta gradient (dCost_dZ) calculations:
    matrix_transpose_into(
        l->next_layer->cache.dense.weight, 
        l->cache.dense.dZnext_dA_t
    ); 
    matrix_dot_into(
        l->next_layer->cache.dense.delta, 
        l->cache.dense.dZnext_dA_t, 
        l->cache.dense.dCost_dA
    );
    apply_activation_dZ_into(
        l->activation, 
        l->cache.dense.z, 
        l->cache.dense.dActivation_dZ
    );
    matrix_multiply_into(
        l->cache.dense.dCost_dA, 
        l->cache.dense.dActivation_dZ, 
        l->cache.dense.delta
    );

    // weight gradient (dCost_dW) calculations:
    matrix_transpose_into(
        layer_get_output_matrix(l->prev_layer), 
        l->cache.dense.dZ_dW_t
    );
    matrix_dot_into(
        l->cache.dense.dZ_dW_t, 
        l->cache.dense.delta, 
        l->cache.dense.weight_gradient
    );

    // bias gradient (dCost_dB) calculations:
    matrix_sum_axis_into(
        l->cache.dense.delta, 
        1, 
        l->cache.dense.dCost_dZ_col_sum
    );
    matrix_multiplicate_into(
        l->cache.dense.dCost_dZ_col_sum, 
        1, 
        net->label_batch->batch_size, 
        l->cache.dense.bias_gradient
    );
}

void layer_conv2D_bp(Layer* l, NeuralNet* net) {
    Tensor4D* delta_next = l->next_layer->cache.conv.delta;
    Tensor4D* filter_next = l->next_layer->cache.conv.filter;
    Tensor4D* output = layer_get_output_tensor4D(l);
    Tensor4D* filter = l->cache.conv.filter;
    Tensor4D* filter_grad = l->cache.conv.filter_gradient;
    Tensor4D* bias_grad = l->cache.conv.bias_gradient;
    Tensor4D* delta = l->cache.conv.delta;
    Tensor4D* dA_dZ = l->cache.conv.dActivation_dZ;
    Tensor4D* dCost_dA = l->cache.conv.dCost_dA;
    Tensor4D* z = l->cache.conv.z;

    // delta gradient (dCost_dZ) calculation
    if (l->next_layer->l_type == FLATTEN) {
        for (int n=0; n<net->batch_size; n++) {
            for (int c=0; c<l->cache.conv.dCost_dA->n_channels; c++) {
                apply_activation_dZ_into(
                    l->activation,
                    z->filters[n]->channels[c],
                    dA_dZ->filters[n]->channels[c]
                );
                matrix_multiply_into(
                    dCost_dA->filters[n]->channels[c],
                    dA_dZ->filters[n]->channels[c],
                    delta->filters[n]->channels[c]
                );
            }
        }
    }
    else if (l->next_layer->l_type == MAX_POOL) {
        Tensor4D* output_next = layer_get_output_tensor4D(l->next_layer);
        int pool_size = l->next_layer->params.conv.filter_size;
        int stride = l->next_layer->params.conv.stride;
        tensor4D_fill(dCost_dA, 0.0);

        for (int n=0; n<net->batch_size; n++) {
            for (int c=0; c<output_next->n_channels; c++) {
                Matrix* out_next_mat = output_next->filters[n]->channels[c];
                Matrix* out_mat = output->filters[n]->channels[c];
                Matrix* dCost_dA_mat = dCost_dA->filters[n]->channels[c];
                Matrix* delta_next_mat = delta_next->filters[n]->channels[c];
                for (int i=0; i<output_next->n_rows; i++) {
                    for (int j=0; j<output_next->n_cols; j++) {
                        for (int k=0; k<pool_size; k++) {
                            for (int l=0; l<pool_size; l++) {
                                if (matrix_get(out_mat, i*stride+k, j*stride+l) ==  matrix_get(out_next_mat, i, j)) {
                                    matrix_assign(
                                        dCost_dA_mat,
                                        i*stride+k,
                                        j*stride+l,
                                        matrix_get(dCost_dA_mat, i*stride+k, j*stride+l) + matrix_get(delta_next_mat, i, j)
                                    );
                                    l = pool_size;
                                    k = pool_size;
                                }
                            }
                        }
                    }
                }

            }
        }

        for (int n=0; n<net->batch_size; n++) {
            for (int c=0; c<l->cache.conv.dCost_dA->n_channels; c++) {
                apply_activation_dZ_into(
                    l->activation,
                    z->filters[n]->channels[c],
                    dA_dZ->filters[n]->channels[c]
                );
                matrix_multiply_into(
                    dCost_dA->filters[n]->channels[c],
                    dA_dZ->filters[n]->channels[c],
                    delta->filters[n]->channels[c]
                );
            }
        }
    }
    else if (l->next_layer->l_type == CONV_2D) {
        Tensor3D* corr_t3d = tensor3D_new(
            delta->n_rows,
            delta->n_cols,
            filter_next->n_filters
        );
        for (int n=0; n<net->batch_size; n++) {
            for (int i=0; i<delta->n_channels; i++) {
                for (int f=0; f<filter_next->n_filters; f++) {
                    matrix_convolve_into(
                        delta_next->filters[n]->channels[f],
                        filter_next->filters[f]->channels[i],
                        corr_t3d->channels[f],
                        l->params.conv.stride,
                        FULL
                    );
                }
                tensor3D_sum_element_wise_into(
                    corr_t3d,
                    dCost_dA->filters[n]->channels[i]
                );
                apply_activation_dZ_into(
                    l->activation,
                    z->filters[n]->channels[i],
                    dA_dZ->filters[n]->channels[i]
                );
                matrix_multiply_into(
                    dCost_dA->filters[n]->channels[i],
                    dA_dZ->filters[n]->channels[i],
                    delta->filters[n]->channels[i]
                );

            }
        }
        tensor3D_free(corr_t3d);
    }


    // filter gradient (dCost_dW) calculation
    {
        Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
        Tensor3D* corr_t3d = tensor3D_new(
            filter->n_rows,
            filter->n_cols,
            net->batch_size
        );
        for (int i=0; i<filter->n_filters; i++) {
            for (int j=0; j<filter->n_channels; j++) {
                for (int n=0; n<net->batch_size; n++) {
                    matrix_correlate_into(
                        input->filters[n]->channels[j],
                        delta->filters[n]->channels[i],
                        corr_t3d->channels[n],
                        l->params.conv.stride,
                        VALID
                    );
                    tensor3D_sum_element_wise_into(
                        corr_t3d,
                        filter_grad->filters[i]->channels[j]
                    );
                }
            }
        }
        tensor3D_free(corr_t3d);
    }

    // bias gradient (dCost_dB) calculation
    for (int i=0; i<filter->n_filters; i++) {
        float sum = 0.0;
        for (int n=0; n<net->batch_size; n++) {
            sum += matrix_sum(delta->filters[n]->channels[i]);
        }
        matrix_fill(bias_grad->filters[i]->channels[0], sum);
    }
}

void layer_flatten_bp(Layer* l, NeuralNet* net) {
    matrix_transpose_into(
        l->next_layer->cache.dense.weight, 
        l->cache.flat.dZnext_dA_t
    ); 
    matrix_dot_into(
        l->next_layer->cache.dense.delta,
        l->cache.flat.dZnext_dA_t,
        l->cache.flat.dCost_dA_matrix
    );
    matrix_into_tensor4D(
        l->cache.flat.dCost_dA_matrix,
        l->prev_layer->cache.conv.dCost_dA
    );
}

void layer_max_pool_bp(Layer* l, NeuralNet* net) {
    // dCost_dZ calculation
    if (l->next_layer->l_type != FLATTEN) {
        Tensor4D* delta = layer_get_delta_tensor4D(l);
        Tensor4D* delta_next = layer_get_delta_tensor4D(l->next_layer);
        Tensor4D* filter_next = l->next_layer->cache.conv.filter;
        Tensor3D* corr_t3d = tensor3D_new(
            delta->n_rows,
            delta->n_cols,
            filter_next->n_filters
        );
        for (int n=0; n<net->batch_size; n++) {
            for (int i=0; i<delta->n_channels; i++) {
                for (int f=0; f<filter_next->n_filters; f++) {
                    matrix_convolve_into(
                        delta_next->filters[n]->channels[f],
                        filter_next->filters[f]->channels[i],
                        corr_t3d->channels[f],
                        l->params.conv.stride,
                        FULL
                    );
                }
                tensor3D_sum_element_wise_into(
                    corr_t3d,
                    delta->filters[n]->channels[i]
                );
            }
        }
    }
}

void layer_deep_update_weights(Layer* l, NeuralNet* net) {
    net->optimizer->update_weights(
        l->cache.dense.weight, 
        l->cache.dense.weight_gradient, 
        net->optimizer, 
        l->layer_idx
    );
    net->optimizer->update_bias(
        l->cache.dense.bias, 
        l->cache.dense.bias_gradient, 
        net->optimizer, 
        l->layer_idx
    );
}

void layer_conv2D_update_weights(Layer* l, NeuralNet* net) {
    Tensor4D* filter = l->cache.conv.filter;
    Tensor4D* bias = l->cache.conv.bias;
    Tensor4D* filter_grad = l->cache.conv.filter_gradient;
    Tensor4D* bias_grad = l->cache.conv.bias_gradient;
    for (int i=0; i<filter->n_filters; i++) {
        for (int j=0; j<filter->n_channels; j++) {
            net->optimizer->update_weights(
                filter->filters[i]->channels[j],
                filter_grad->filters[i]->channels[j],
                net->optimizer,
                l->layer_idx
            );
        }
        net->optimizer->update_bias(
            bias->filters[i]->channels[0],
            bias_grad->filters[i]->channels[0],
            net->optimizer,
            l->layer_idx
        );
    }
    kernel_into_im2col(filter, false, l->cache.conv.kernel_im2col);
}