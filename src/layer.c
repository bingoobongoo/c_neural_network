#include "layer.h"

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
        if (layer->cache.conv.fp_im2col_input != NULL) {
            matrix_free(layer->cache.conv.fp_im2col_input);
            layer->cache.conv.fp_im2col_input = NULL;
        }
        if (layer->cache.conv.fp_im2col_kernel != NULL) {
            matrix_free(layer->cache.conv.fp_im2col_kernel);
            layer->cache.conv.fp_im2col_kernel = NULL;
        }
        if (layer->cache.conv.fp_im2col_output != NULL) {
            matrix_free(layer->cache.conv.fp_im2col_output);
            layer->cache.conv.fp_im2col_output = NULL;
        }
        if (layer->cache.conv.dCost_dW_im2col_input != NULL) {
            matrix_free(layer->cache.conv.dCost_dW_im2col_input);
            layer->cache.conv.dCost_dW_im2col_input = NULL;
        }
        if (layer->cache.conv.dCost_dW_im2col_kernel != NULL) {
            matrix_free(layer->cache.conv.dCost_dW_im2col_kernel);
            layer->cache.conv.dCost_dW_im2col_kernel = NULL;
        }
        if (layer->cache.conv.dCost_dW_im2col_output != NULL) {
            matrix_free(layer->cache.conv.dCost_dW_im2col_output);
            layer->cache.conv.dCost_dW_im2col_output = NULL;
        }
        if (layer->cache.conv.delta_im2col_input != NULL) {
            matrix_free(layer->cache.conv.delta_im2col_input);
            layer->cache.conv.delta_im2col_input = NULL;
        }
        if (layer->cache.conv.delta_im2col_kernel != NULL) {
            matrix_free(layer->cache.conv.delta_im2col_kernel);
            layer->cache.conv.delta_im2col_kernel = NULL;
        }
        if (layer->cache.conv.delta_im2col_output != NULL) {
            matrix_free(layer->cache.conv.delta_im2col_output);
            layer->cache.conv.delta_im2col_output = NULL;
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

void layer_deep_compile(Layer* l, ActivationType act_type, int act_param, int batch_size) {
    l->activation = activation_new(
        act_type,
        act_param
    );
    l->cache.dense.weight = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.output = matrix_new(
        batch_size, 
        layer_get_n_units(l)
    );
    l->cache.dense.z = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.delta = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.weight_gradient = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias_gradient = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dCost_dA = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dActivation_dZ = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dZ_dW_t = matrix_new(
        layer_get_n_units(l->prev_layer),
        batch_size
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

    switch (act_type)
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

void layer_output_compile(Layer* l, Cost* cost, int batch_size) {
    if (layer_get_n_units(l) == 1)
        l->activation = activation_new(SIGMOID, 0.0);
    if (layer_get_n_units(l) > 1)
        l->activation = activation_new(SOFTMAX, 0.0);
    l->activation->y_true_batch = batch_matrix_new(
        batch_size,
        layer_get_n_units(l)
    );

    cost->loss_m = matrix_new(batch_size, layer_get_n_units(l));

    l->cache.dense.weight = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.output = matrix_new(
        batch_size, 
        layer_get_n_units(l)
    );
    l->cache.dense.z = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.delta = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.weight_gradient = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias_gradient = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dCost_dA = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dActivation_dZ = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dZ_dW_t = matrix_new(
        layer_get_n_units(l->prev_layer),
        batch_size
    );
    l->cache.dense.dCost_dZ_col_sum = matrix_new(
        1,
        layer_get_n_units(l)
    );
    l->cache.dense.dZnext_dA_t = NULL;

    matrix_fill(l->cache.dense.bias, 0.0);
    switch (l->activation->type)
    {
    case SIGMOID:
    case SOFTMAX:
        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            0.0,
            2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer))
        );
        break;

    default:
        printf("Unknown activation type.");
        exit(1);
        break;
    }
}

void layer_conv2D_compile(Layer* l, ActivationType act_type, int act_param, int batch_size) {
    int input_height = layer_get_output_tensor4D(l->prev_layer)->n_rows;
    int input_width = layer_get_output_tensor4D(l->prev_layer)->n_cols;
    int input_channels = layer_get_output_tensor4D(l->prev_layer)->n_channels;
    int filter_height = l->params.conv.filter_size;
    int filter_width = l->params.conv.filter_size;
    int stride = l->params.conv.stride;
    int output_height = floor((input_height - filter_height) / stride) + 1;
    int output_width = floor((input_width - filter_width) / stride) + 1;

    l->activation = activation_new(
        act_type,
        act_param
    );
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
        batch_size
    );
    l->cache.conv.z = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        batch_size
    );
    l->cache.conv.delta = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        batch_size
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
        batch_size
    );
    l->cache.conv.dActivation_dZ = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        batch_size
    );
    l->cache.conv.fp_im2col_input = matrix_new(
        output_height * output_width,
        filter_height * filter_width * input_channels
    );
    l->cache.conv.fp_im2col_kernel = matrix_new(
        filter_height * filter_width * input_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.fp_im2col_output = matrix_new(
        output_height * output_width,
        l->params.conv.n_filters
    );
    l->cache.conv.dCost_dW_im2col_input = matrix_new(
        filter_height * filter_width,
        output_height * output_width * batch_size
    );
    l->cache.conv.dCost_dW_im2col_kernel = matrix_new(
        output_height * output_width * batch_size,
        l->params.conv.n_filters
    );
    l->cache.conv.dCost_dW_im2col_output = matrix_new(
        filter_height * filter_width,
        l->params.conv.n_filters
    );
    l->cache.conv.delta_im2col_input = matrix_new(
        input_height * input_width,
        filter_height * filter_width * l->params.conv.n_filters
    );
    l->cache.conv.delta_im2col_kernel = matrix_new(
        filter_height * filter_width * l->params.conv.n_filters,
        input_channels
    );
    l->cache.conv.delta_im2col_output = matrix_new(
        input_height * input_width,
        input_channels
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
    
    kernel_into_im2col_fwise(
        l->cache.conv.filter,
        false,
        l->cache.conv.fp_im2col_kernel
    );
}

void layer_flatten_compile(Layer* l, int batch_size) {
    int prev_chan = layer_get_output_tensor4D(l->prev_layer)->n_channels;
    int prev_rows = layer_get_output_tensor4D(l->prev_layer)->n_rows;
    int prev_cols = layer_get_output_tensor4D(l->prev_layer)->n_cols;
    l->params.flat.n_units = prev_chan * prev_rows * prev_cols;
    l->cache.flat.output = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.flat.dCost_dA_matrix = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.flat.dZnext_dA_t = matrix_new(
        layer_get_n_units(l->next_layer),
        layer_get_n_units(l)
    );
}

void layer_max_pool_compile(Layer* l, int batch_size) {
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
        batch_size
    );
    l->cache.conv.delta = tensor4D_new(
        output_height,
        output_width,
        input_channels,
        batch_size
    );
    l->cache.conv.dCost_dA = l->cache.conv.delta;

    l->cache.conv.filter = NULL;
    l->cache.conv.bias = NULL;
    l->cache.conv.z = NULL;
    l->cache.conv.filter_gradient = NULL;
    l->cache.conv.bias_gradient = NULL;
    l->cache.conv.dActivation_dZ = NULL;
    l->cache.conv.fp_im2col_input = NULL;
    l->cache.conv.fp_im2col_kernel = NULL;
    l->cache.conv.fp_im2col_output = NULL;
    l->cache.conv.dCost_dW_im2col_input = NULL;
    l->cache.conv.dCost_dW_im2col_kernel = NULL;
    l->cache.conv.dCost_dW_im2col_output = NULL;

    l->params.conv.n_units = 
        l->cache.conv.output->n_rows *
        l->cache.conv.output->n_cols *
        l->cache.conv.output->n_channels;
}

void layer_input_fp(Layer* l, Batch* train_batch, int batch_size) {
    l->cache.dense.output = train_batch->data.matrix;
}

void layer_conv2D_input_fp(Layer* l, Batch* train_batch, int batch_size) {
    l->cache.conv.output = train_batch->data.tensor;
}

void layer_deep_fp(Layer* l, int batch_size) {
    matrix_dot_into(layer_get_output_matrix(l->prev_layer), l->cache.dense.weight, l->cache.dense.z);
    matrix_add_into(l->cache.dense.z, l->cache.dense.bias, l->cache.dense.z);
    apply_activation_func_into(l->activation, l->cache.dense.z, l->cache.dense.output);
}

void layer_output_fp(Layer* l, Batch* label_batch, int batch_size) {
    matrix_dot_into(layer_get_output_matrix(l->prev_layer), l->cache.dense.weight, l->cache.dense.z);
    matrix_add_into(l->cache.dense.z, l->cache.dense.bias, l->cache.dense.z);
    l->activation->y_true_batch = label_batch;
    apply_activation_func_into(l->activation, l->cache.dense.z, l->cache.dense.output);
}

void layer_conv2D_fp(Layer* l, int batch_size) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* filter = l->cache.conv.filter;
    Tensor4D* z = l->cache.conv.z;
    Tensor4D* bias = l->cache.conv.bias;
    Tensor4D* output = l->cache.conv.output;
    
    for (int n=0; n<batch_size; n++) {
        input_into_im2col_fwise(
            input,
            n,
            filter,
            l->params.conv.stride,
            VALID,
            l->cache.conv.fp_im2col_input 
        );
        matrix_dot_into(
            l->cache.conv.fp_im2col_input,
            l->cache.conv.fp_im2col_kernel,
            l->cache.conv.fp_im2col_output
        );
        matrix_into_tensor3D(
            l->cache.conv.fp_im2col_output,
            z->filters[n],
            true
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
    // for (int n=0; n<batch_size; n++) {
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

void layer_flatten_fp(Layer* l, int batch_size) {
    tensor4D_into_matrix_fwise(
        layer_get_output_tensor4D(l->prev_layer),
        l->cache.flat.output,
        false,
        false
    );
}

void layer_max_pool_fp(Layer* l, int batch_size) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* output = layer_get_output_tensor4D(l);

    for (int n=0; n<batch_size; n++) {
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

void layer_output_bp(Layer* l, Cost* cost, Batch* label_batch, int batch_size) {
    // delta gradient (dCost_dZ) calculations:
    apply_cost_dA_into(
        cost, 
        l->cache.dense.output, 
        label_batch->data.matrix, 
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
        batch_size, 
        l->cache.dense.bias_gradient
    );
}

void layer_deep_bp(Layer* l, int batch_size) {
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
        batch_size, 
        l->cache.dense.bias_gradient
    );
}

void layer_conv2D_bp(Layer* l, int batch_size) {
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
        for (int n=0; n<batch_size; n++) {
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

        for (int n=0; n<batch_size; n++) {
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

        for (int n=0; n<batch_size; n++) {
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
        kernel_into_im2col_chwise(
            filter_next,
            true,
            l->next_layer->cache.conv.delta_im2col_kernel
        );

        for (int n=0; n<batch_size; n++) {
            input_into_im2col_fwise(
                delta_next,
                n,
                filter_next,
                l->next_layer->params.conv.stride,
                FULL,
                l->next_layer->cache.conv.delta_im2col_input
            );
            matrix_dot_into(
                l->next_layer->cache.conv.delta_im2col_input,
                l->next_layer->cache.conv.delta_im2col_kernel,
                l->next_layer->cache.conv.delta_im2col_output
            );
            matrix_into_tensor3D(
                l->next_layer->cache.conv.delta_im2col_output,
                dCost_dA->filters[n],
                true
            );
            for (int i=0; i<delta->n_channels; i++) {
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
        // Tensor3D* corr_t3d = tensor3D_new(
        //     delta->n_rows,
        //     delta->n_cols,
        //     filter_next->n_filters
        // );
        // for (int n=0; n<batch_size; n++) {
        //     for (int i=0; i<delta->n_channels; i++) {
        //         for (int f=0; f<filter_next->n_filters; f++) {
        //             matrix_convolve_into(
        //                 delta_next->filters[n]->channels[f],
        //                 filter_next->filters[f]->channels[i],
        //                 corr_t3d->channels[f],
        //                 l->params.conv.stride,
        //                 FULL
        //             );
        //         }
        //         tensor3D_sum_element_wise_into(
        //             corr_t3d,
        //             dCost_dA->filters[n]->channels[i]
        //         );
        //         apply_activation_dZ_into(
        //             l->activation,
        //             z->filters[n]->channels[i],
        //             dA_dZ->filters[n]->channels[i]
        //         );
        //         matrix_multiply_into(
        //             dCost_dA->filters[n]->channels[i],
        //             dA_dZ->filters[n]->channels[i],
        //             delta->filters[n]->channels[i]
        //         );

        //     }
        // }
        // tensor3D_free(corr_t3d);
    }


    // filter gradient (dCost_dW) calculation
    {
        Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
        
        kernel_into_im2col_chwise(
            delta,
            false,
            l->cache.conv.dCost_dW_im2col_kernel
        );
        for (int i=0; i<filter->n_channels; i++) {
            input_into_im2col_chwise(
                input,
                i,
                delta,
                l->params.conv.stride,
                VALID,
                l->cache.conv.dCost_dW_im2col_input
            );
            matrix_dot_into(
                l->cache.conv.dCost_dW_im2col_input,
                l->cache.conv.dCost_dW_im2col_kernel,
                l->cache.conv.dCost_dW_im2col_output
            );
            
            int n_out_rows = l->cache.conv.dCost_dW_im2col_output->n_rows;
            int n_out_cols = l->cache.conv.dCost_dW_im2col_output->n_cols;
            for (int a=0; a<n_out_rows; a++) {
                for (int b=0; b<n_out_cols; b++) {
                    matrix_assign(
                        filter_grad->filters[b]->channels[i],
                        a / filter->n_cols,
                        a % filter->n_cols,
                        matrix_get(
                            l->cache.conv.dCost_dW_im2col_output,
                            a,
                            b
                        )
                    );
                }
            }
        }
        // Tensor3D* corr_t3d = tensor3D_new(
        //     filter->n_rows,
        //     filter->n_cols,
        //     batch_size
        // );
        // for (int i=0; i<filter->n_filters; i++) {
        //     for (int j=0; j<filter->n_channels; j++) {
        //         for (int n=0; n<batch_size; n++) {
        //             matrix_correlate_into(
        //                 input->filters[n]->channels[j],
        //                 delta->filters[n]->channels[i],
        //                 corr_t3d->channels[n],
        //                 l->params.conv.stride,
        //                 VALID
        //             );
        //         }
        //         tensor3D_sum_element_wise_into(
        //             corr_t3d,
        //             filter_grad->filters[i]->channels[j]
        //         );
        //     }
        // }
        // tensor3D_free(corr_t3d);
    }

    // bias gradient (dCost_dB) calculation
    for (int i=0; i<filter->n_filters; i++) {
        nn_float sum = 0.0;
        for (int n=0; n<batch_size; n++) {
            sum += matrix_sum(delta->filters[n]->channels[i]);
        }
        matrix_fill(bias_grad->filters[i]->channels[0], sum);
    }
}

void layer_flatten_bp(Layer* l, int batch_size) {
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

void layer_max_pool_bp(Layer* l, int batch_size) {
    // dCost_dZ calculation
    if (l->next_layer->l_type != FLATTEN) {
        Tensor4D* delta = layer_get_delta_tensor4D(l);
        Tensor4D* delta_next = layer_get_delta_tensor4D(l->next_layer);
        Tensor4D* filter_next = l->next_layer->cache.conv.filter;

        kernel_into_im2col_chwise(
            filter_next,
            true,
            l->next_layer->cache.conv.delta_im2col_kernel
        );

        for (int n=0; n<batch_size; n++) {
            input_into_im2col_fwise(
                delta_next,
                n,
                filter_next,
                l->next_layer->params.conv.stride,
                FULL,
                l->next_layer->cache.conv.delta_im2col_input
            );
            matrix_dot_into(
                l->next_layer->cache.conv.delta_im2col_input,
                l->next_layer->cache.conv.delta_im2col_kernel,
                l->next_layer->cache.conv.delta_im2col_output
            );
            matrix_into_tensor3D(
                l->next_layer->cache.conv.delta_im2col_output,
                delta->filters[n],
                true
            );
        }
        // Tensor3D* corr_t3d = tensor3D_new(
        //     delta->n_rows,
        //     delta->n_cols,
        //     filter_next->n_filters
        // );
        // for (int n=0; n<batch_size; n++) {
        //     for (int i=0; i<delta->n_channels; i++) {
        //         for (int f=0; f<filter_next->n_filters; f++) {
        //             matrix_convolve_into(
        //                 delta_next->filters[n]->channels[f],
        //                 filter_next->filters[f]->channels[i],
        //                 corr_t3d->channels[f],
        //                 l->params.conv.stride,
        //                 FULL
        //             );
        //         }
        //         tensor3D_sum_element_wise_into(
        //             corr_t3d,
        //             delta->filters[n]->channels[i]
        //         );
        //     }
        // }
    }
}

void layer_deep_update_weights(Layer* l, Optimizer* opt) {
    opt->update_weights(
        l->cache.dense.weight, 
        l->cache.dense.weight_gradient, 
        opt, 
        l->layer_idx
    );
    opt->update_bias(
        l->cache.dense.bias, 
        l->cache.dense.bias_gradient, 
        opt, 
        l->layer_idx
    );
}

void layer_conv2D_update_weights(Layer* l, Optimizer* opt) {
    Tensor4D* filter = l->cache.conv.filter;
    Tensor4D* bias = l->cache.conv.bias;
    Tensor4D* filter_grad = l->cache.conv.filter_gradient;
    Tensor4D* bias_grad = l->cache.conv.bias_gradient;
    for (int i=0; i<filter->n_filters; i++) {
        for (int j=0; j<filter->n_channels; j++) {
            opt->update_weights(
                filter->filters[i]->channels[j],
                filter_grad->filters[i]->channels[j],
                opt,
                l->layer_idx
            );
        }
        opt->update_bias(
            bias->filters[i]->channels[0],
            bias_grad->filters[i]->channels[0],
            opt,
            l->layer_idx
        );
    }

    kernel_into_im2col_fwise(filter, false, l->cache.conv.fp_im2col_kernel);
}