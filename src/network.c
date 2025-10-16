#include "network.h"

NeuralNet* neural_net_new(Optimizer* opt, ActivationType act_type, nn_float act_param, CostType cost_type, int batch_size) {
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
            layer_deep_compile(
                layer,
                net->activation->type,
                net->activation->activation_param,
                net->batch_size
            );
            break;

        case OUTPUT: 
            layer_output_compile(
                layer,
                net->cost,
                net->batch_size
            );
            break;

        case CONV_2D: 
            layer_conv2D_compile(
                layer,
                net->activation->type,
                net->activation->activation_param,
                net->batch_size
            );
            break;

        case FLATTEN: 
            layer_flatten_compile(layer, net->batch_size);
            break;

        case MAX_POOL: 
            layer_max_pool_compile(layer, net->batch_size);
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
            matrix_fill(mom->weight_momentum[i], (nn_float)0.0);

            mom->bias_momentum[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            matrix_fill(mom->bias_momentum[i], (nn_float)0.0);
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
            matrix_fill(ada->weight_s[i], (nn_float)0.0);

            ada->bias_s[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            matrix_fill(ada->bias_s[i], (nn_float)0.0);

            ada->intermediate_w[i] = matrix_new(
                layer_get_n_units(layer->prev_layer), 
                layer_get_n_units(layer)
            );
            matrix_fill(ada->intermediate_w[i], (nn_float)0.0);

            ada->intermediate_b[i] = matrix_new(
                net->batch_size, 
                layer_get_n_units(layer)
            );
            matrix_fill(ada->intermediate_b[i], (nn_float)0.0);
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

            matrix_fill(adam->weight_m[i], (nn_float)0.0);
            matrix_fill(adam->weight_m_corr[i], (nn_float)0.0);
            matrix_fill(adam->weight_s[i], (nn_float)0.0);
            matrix_fill(adam->weight_s_corr[i], (nn_float)0.0);
            matrix_fill(adam->intermediate_w[i], (nn_float)0.0);
            
            adam->bias_m[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            adam->bias_m_corr[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            adam->bias_s[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            adam->bias_s_corr[i] = matrix_new(net->batch_size, layer_get_n_units(layer));
            adam->intermediate_b[i] = matrix_new(net->batch_size, layer_get_n_units(layer));

            matrix_fill(adam->bias_m[i], (nn_float)0.0);
            matrix_fill(adam->bias_m_corr[i], (nn_float)0.0);
            matrix_fill(adam->bias_s[i], (nn_float)0.0);
            matrix_fill(adam->bias_s_corr[i], (nn_float)0.0);
            matrix_fill(adam->intermediate_b[i], (nn_float)0.0);
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
        printf("------------------------------------------\n");
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
                printf("------------------------------------------\n");
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
                printf("------------------------------------------\n");
                break;
            }
            case OUTPUT: {
                l_name = "Output";
                params = n_units * layer_get_n_units(layer->prev_layer) + n_units;
                trainable_params += params;
                printf("%d  %s  %d  (%d x %d)  %lld\n", i, l_name, n_units, batch_size, n_units, params);
                printf("------------------------------------------\n");
                break;
            }
            case DEEP: {
                l_name = "Deep";
                params = n_units * layer_get_n_units(layer->prev_layer) + n_units;
                trainable_params += params;
                printf("%d  %s  %d  (%d x %d)  %lld\n", i, l_name, n_units, batch_size, n_units, params);
                printf("------------------------------------------\n");
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
                printf("------------------------------------------\n");
                break;
            }
            case FLATTEN: {
                l_name = "Flatten";
                params = 0;
                printf("%d  %s  %d  (%d x %d)  %lld\n", i, l_name, n_units, batch_size, n_units, params);
                printf("------------------------------------------\n");
                break;
            }
            case MAX_POOL: {
                int out_chan = layer->cache.conv.output->n_channels;
                int out_rows = layer->cache.conv.output->n_rows;
                int out_cols = layer->cache.conv.output->n_cols;
                l_name = "MaxPool";
                params = 0;
                printf("%d  %s  %d  (%d x %d x %d x %d)  %lld\n", i, l_name, n_units, batch_size, out_chan, out_rows, out_cols, params);
                printf("------------------------------------------\n");
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

void fit(Matrix* x_train, Matrix* y_train, int n_epochs, nn_float validation, NeuralNet* net) {
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

        train_batches = ceil(x_train_split_tensor->n_filters / (nn_float)net->batch_size);
        val_batches = ceil(x_val_split_tensor->n_filters / (nn_float)net->batch_size);
        
        matrix_free(x_train_split_mat);
        matrix_free(x_val_split_mat);
    }
    else {
        train_batches = ceil(x_train_split_mat->n_rows / (nn_float)net->batch_size);
        val_batches = ceil(x_val_split_mat->n_rows / (nn_float)net->batch_size); 
    }
    
    Matrix* y_train_split = matrix_slice_rows(y_train, 0, training_size);
    Matrix* y_val_split = matrix_slice_rows(y_train, training_size, val_size);

    nn_float* avg_loss = (nn_float*)malloc(train_batches * sizeof(nn_float));
    nn_float* train_acc = (nn_float*)malloc(train_batches * sizeof(nn_float));
    nn_float* val_acc = (nn_float*)malloc(val_batches * sizeof(nn_float));

    int start_idx = 0;
    int i = 0;

    nn_float epoch_time;
    nn_float sum;
    nn_float avg_epoch_loss;
    nn_float avg_epoch_train_acc;
    nn_float avg_epoch_val_acc;

    for (int epoch=0; epoch<n_epochs; epoch++) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (start_idx=0, i=0; start_idx<training_size - net->batch_size; start_idx+=net->batch_size, i++) {
            if (net->is_cnn) {
                batchify_tensor_into(x_train_split_tensor, start_idx, net->train_batch);
            }
            else {
                batchify_matrix_into(x_train_split_mat, start_idx, net->train_batch);
            }
            batchify_matrix_into(y_train_split, start_idx, net->label_batch);
            forward_prop(net, true);
            avg_loss[i] = get_avg_batch_loss(
                net->cost, 
                net->layers[net->n_layers-1]->cache.dense.output, 
                net->label_batch->data.matrix
            );
            Matrix* y_pred = net->layers[net->n_layers-1]->cache.dense.output;
            Matrix* y_true = net->label_batch->data.matrix;
            score_batch(net->batch_score, y_pred, y_true);
            train_acc[i] = net->batch_score->accuracy;
            back_prop(net);
            #ifdef DEBUG
            if (i==0) {
                printf("Stats after 1 batch:\n");
                debug_layers_info(net);
            }
            #endif
        }
        gettimeofday(&end, NULL);
        epoch_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;;

        sum = (nn_float)0.0;
        for (int j=0; j<i; j++) {
            sum += avg_loss[j];
        }
        avg_epoch_loss = sum / (nn_float)i;

        sum = (nn_float)0.0;
        for (int j=0; j<i; j++) {
            sum += train_acc[j];
        }
        avg_epoch_train_acc = sum / (nn_float)i;

        for (start_idx=0, i=0; start_idx<val_size - net->batch_size; start_idx+=net->batch_size, i++) {
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

        sum = (nn_float)0.0;
        for (int j=0; j<i; j++) {
            sum += val_acc[j];
        }
        avg_epoch_val_acc = sum / (nn_float)i;

        if (net->is_cnn) {
            shuffle_tensor4D_inplace(x_train_split_tensor, y_train_split);
        }
        else {
            shuffle_matrix_inplace(x_train_split_mat, y_train_split);
        }

        printf("Epoch: %d/%d   loss: %f   train_acc: %.4f   val_acc: %.4f   time: %.3fs\n", epoch+1, n_epochs, avg_epoch_loss, avg_epoch_train_acc, avg_epoch_val_acc, epoch_time);

        #ifdef DEBUG
        debug_layers_info(net);
        #endif
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
        n_batches = ceil(x_test_tensor->n_filters / (nn_float)net->batch_size);
    }
    else {
        n_batches = ceil(x_test->n_rows / (nn_float)net->batch_size);
    }

    int start_idx = 0;
    int i = 0;
    nn_float* acc = (nn_float*)malloc(n_batches * sizeof(nn_float));

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

    nn_float sum = (nn_float)0.0;
    for (int j=0; j<i; j++) {
        sum += acc[j];
    }
    nn_float test_acc = sum / (nn_float)i;

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
    matrix_fill(conf_m, (nn_float)0.0);

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
            layer_input_fp(
                l,
                input,
                net->batch_size
            );
            break;
        
        case CONV_2D_INPUT:
            layer_conv2D_input_fp(
                l,
                input,
                net->batch_size
            );
            break;

        case DEEP:
            if (net->optimizer->type == NESTEROV && training) {
                    MomentumConfig* mom = (MomentumConfig*)net->optimizer->settings;
                    matrix_add_into(l->cache.dense.weight, mom->weight_momentum[i], l->cache.dense.weight);
                    matrix_add_into(l->cache.dense.bias, mom->bias_momentum[i], l->cache.dense.bias);
                }
                layer_deep_fp(l, net->batch_size);
                break;
        case OUTPUT:
            if (net->optimizer->type == NESTEROV && training) {
                MomentumConfig* mom = (MomentumConfig*)net->optimizer->settings;
                matrix_add_into(l->cache.dense.weight, mom->weight_momentum[i], l->cache.dense.weight);
                matrix_add_into(l->cache.dense.bias, mom->bias_momentum[i], l->cache.dense.bias);
            }
            layer_output_fp(l, net->label_batch, net->batch_size);
            break;

        case CONV_2D:
            layer_conv2D_fp(l, net->batch_size);
            break;
            
        case FLATTEN:
            layer_flatten_fp(l, net->batch_size);
            break;
        
        case MAX_POOL:
            layer_max_pool_fp(l, net->batch_size);
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
            layer_output_bp(
                l,
                net->cost,
                net->label_batch,
                net->batch_size
            );
            break;
        
        case DEEP:
            layer_deep_bp(l, net->batch_size);
            break;

        case CONV_2D:
            layer_conv2D_bp(l, net->batch_size);
            break;

        case FLATTEN:
            layer_flatten_bp(l, net->batch_size);
            break;

        case MAX_POOL: 
            layer_max_pool_bp(l, net->batch_size);
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
                    layer_deep_update_weights(l, net->optimizer);
                    break;

                case CONV_2D:
                    layer_conv2D_update_weights(l, net->optimizer);
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

void debug_layers_info(NeuralNet* net) {
    printf("lp  layer   min_grad   max_grad   mean_grad   min_weight   max_weight   mean_weight");
    printf("   min_act   max_act   mean_act\n");
    printf("--------------------------------------------------------------------------------------------------------------------\n");
    nn_float min_grad, max_grad, mean_grad;
    nn_float min_weight, max_weight, mean_weight;
    nn_float min_act, max_act, mean_act;
    char* l_name = NULL;
    for (int i=0; i<net->n_layers; i++) {
        Layer* layer = net->layers[i];
        switch (layer->l_type)
        {
        case DEEP: {
            l_name = "Dense";
            min_grad = matrix_min(layer->cache.dense.weight_gradient);
            max_grad = matrix_max(layer->cache.dense.weight_gradient);
            mean_grad = matrix_average(layer->cache.dense.weight_gradient);
            min_weight = matrix_min(layer->cache.dense.weight);
            max_weight = matrix_max(layer->cache.dense.weight);
            mean_weight = matrix_average(layer->cache.dense.weight);
            min_act = matrix_min(layer->cache.dense.output);
            max_act = matrix_max(layer->cache.dense.output);
            mean_act = matrix_average(layer->cache.dense.output);
            printf("%d   %s   %f   %f   %f   %f   %f   %f   %f   %f   %f\n", i, l_name, min_grad, max_grad, mean_grad, min_weight, max_weight, mean_weight, min_act, max_act, mean_act);
            printf("--------------------------------------------------------------------------------------------------------------------\n");
            break;
        }
        case OUTPUT: {
            l_name = "Output";
            min_grad = matrix_min(layer->cache.dense.weight_gradient);
            max_grad = matrix_max(layer->cache.dense.weight_gradient);
            mean_grad = matrix_average(layer->cache.dense.weight_gradient);
            min_weight = matrix_min(layer->cache.dense.weight);
            max_weight = matrix_max(layer->cache.dense.weight);
            mean_weight = matrix_average(layer->cache.dense.weight);
            min_act = matrix_min(layer->cache.dense.output);
            max_act = matrix_max(layer->cache.dense.output);
            mean_act = matrix_average(layer->cache.dense.output);
            printf("%d   %s   %f   %f   %f   %f   %f   %f   %f   %f   %f\n", i, l_name, min_grad, max_grad, mean_grad, min_weight, max_weight, mean_weight, min_act, max_act, mean_act);
            printf("--------------------------------------------------------------------------------------------------------------------\n");
            break;            
        }
        }
    }
}