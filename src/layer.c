#include "layer.h"

Layer* layer_new(LayerType l_type, NeuralNet* net) {
    Layer* l = (Layer*)malloc(sizeof(Layer));
    l->l_type = l_type;
    l->activation = NULL;
    l->prev_layer = NULL;
    l->next_layer = NULL;
    l->net_backref = net;

    return l;
}

void layer_free(Layer* l) { 
    switch (l->l_type)
    {
    case INPUT:
    case CONV2D_INPUT:
        free(l);
        return;
        break;
    
    case CONV2D:
        if (l->cache.conv.output != NULL) {
            tensor4D_free(l->cache.conv.output);
            l->cache.conv.output = NULL;
        }
        if (l->cache.conv.z != NULL) {
            tensor4D_free(l->cache.conv.z);
            l->cache.conv.z = NULL;
        }
        if (l->cache.conv.weight != NULL) {
            tensor4D_free(l->cache.conv.weight);
            l->cache.conv.weight = NULL;
        }
        if (l->cache.conv.weight_flip != NULL) {
            tensor4D_free(l->cache.conv.weight_flip);
            l->cache.conv.weight_flip = NULL;
        }
        if (l->cache.conv.bias != NULL) {
            matrix_free(l->cache.conv.bias);
            l->cache.conv.bias = NULL;
        }
        if (l->cache.conv.delta != NULL) {
            tensor4D_free(l->cache.conv.delta);
            l->cache.conv.delta = NULL;
        }
        if (l->cache.conv.weight_grad != NULL) {
            tensor4D_free(l->cache.conv.weight_grad);
            l->cache.conv.weight_grad = NULL;
        }
        if (l->cache.conv.bias_grad != NULL) {
            matrix_free(l->cache.conv.bias_grad);
            l->cache.conv.bias_grad = NULL;
        }
        if (l->cache.conv.dL_dA != NULL) {
            tensor4D_free(l->cache.conv.dL_dA);
            l->cache.conv.dL_dA = NULL;
        }
        if (l->cache.conv.dA_dZ != NULL) {
            tensor4D_free(l->cache.conv.dA_dZ);
            l->cache.conv.dA_dZ = NULL;
        }
        if (l->cache.conv.padding != NULL) {
            tensor4D_free(l->cache.conv.padding);
            l->cache.conv.padding = NULL;
        }
        if (l->cache.conv.fp_im2col_input != NULL) {
            tensor3D_free(l->cache.conv.fp_im2col_input);
            l->cache.conv.fp_im2col_input = NULL;
        }
        if (l->cache.conv.fp_im2col_kernel != NULL) {
            matrix_free(l->cache.conv.fp_im2col_kernel);
            l->cache.conv.fp_im2col_kernel = NULL;
        }
        if (l->cache.conv.fp_im2col_output != NULL) {
            tensor3D_free(l->cache.conv.fp_im2col_output);
            l->cache.conv.fp_im2col_output = NULL;
        }
        if (l->cache.conv.dL_dW_im2col_input != NULL) {
            tensor3D_free(l->cache.conv.dL_dW_im2col_input);
            l->cache.conv.dL_dW_im2col_input = NULL;
        }
        if (l->cache.conv.dL_dW_im2col_kernel != NULL) {
            tensor3D_free(l->cache.conv.dL_dW_im2col_kernel);
            l->cache.conv.dL_dW_im2col_kernel = NULL;
        }
        if (l->cache.conv.dL_dW_im2col_output != NULL) {
            tensor3D_free(l->cache.conv.dL_dW_im2col_output);
            l->cache.conv.dL_dW_im2col_output = NULL;
        }
        if (l->cache.conv.dL_dW_im2col_output_sum != NULL) {
            matrix_free(l->cache.conv.dL_dW_im2col_output_sum);
            l->cache.conv.dL_dW_im2col_output_sum = NULL;
        }
        if (l->cache.conv.delta_im2col_input != NULL) {
            tensor3D_free(l->cache.conv.delta_im2col_input);
            l->cache.conv.delta_im2col_input = NULL;
        }
        if (l->cache.conv.delta_im2col_kernel != NULL) {
            matrix_free(l->cache.conv.delta_im2col_kernel);
            l->cache.conv.delta_im2col_kernel = NULL;
        }
        if (l->cache.conv.delta_im2col_output != NULL) {
            tensor3D_free(l->cache.conv.delta_im2col_output);
            l->cache.conv.delta_im2col_output = NULL;
        }
        break;
        
    case DENSE:
    case OUTPUT:
        if (l->cache.dense.output != NULL) {
            matrix_free(l->cache.dense.output);
            l->cache.dense.output = NULL;
        }
        if (l->cache.dense.z != NULL) {
            matrix_free(l->cache.dense.z);
            l->cache.dense.z = NULL;
        }
        if (l->cache.dense.weight != NULL) {
            matrix_free(l->cache.dense.weight);
            l->cache.dense.weight = NULL;
        }
        if (l->cache.dense.bias != NULL) {
            matrix_free(l->cache.dense.bias);
            l->cache.dense.bias = NULL;
        }
        if (l->cache.dense.delta != NULL) {
            matrix_free(l->cache.dense.delta);
            l->cache.dense.delta = NULL;
        }
        if (l->cache.dense.weight_grad != NULL) {
            matrix_free(l->cache.dense.weight_grad);
            l->cache.dense.weight_grad = NULL;
        }
        if (l->cache.dense.bias_grad != NULL) {
            matrix_free(l->cache.dense.bias_grad);
            l->cache.dense.bias_grad = NULL;
        }
        if (l->cache.dense.dL_dA != NULL) {
            matrix_free(l->cache.dense.dL_dA);
            l->cache.dense.dL_dA = NULL;
        }
        if (l->cache.dense.dA_dZ != NULL) {
            matrix_free(l->cache.dense.dA_dZ);
            l->cache.dense.dA_dZ = NULL;
        }
        break;

    case FLATTEN:
        if (l->cache.flat.output != NULL) {
            matrix_free(l->cache.flat.output);
            l->cache.flat.output = NULL;
        }
        if (l->cache.flat.delta != NULL) {
            matrix_free(l->cache.flat.delta);
            l->cache.flat.delta = NULL;
        }
        break;

    case MAX_POOL: 
        if (l->cache.max_pool.output != NULL) {
            tensor4D_free(l->cache.max_pool.output);
            l->cache.max_pool.output = NULL;
        }
        if (l->cache.max_pool.delta != NULL) {
            tensor4D_free(l->cache.max_pool.delta);
            l->cache.max_pool.delta = NULL;
        }
        if (l->cache.max_pool.argmax != NULL) {
            tensor4D_uint16_free(l->cache.max_pool.argmax);
            l->cache.max_pool.argmax = NULL;
        }
        break;

    case BATCH_NORM_CONV2D:
        if (l->cache.bn_conv.output != NULL) {
            tensor4D_free(l->cache.bn_conv.output);
            l->cache.bn_conv.output = NULL;
        }
        if (l->cache.bn_conv.z != NULL) {
            tensor4D_free(l->cache.bn_conv.z);
            l->cache.bn_conv.z = NULL;
        }
        if (l->cache.bn_conv.delta != NULL) {
            tensor4D_free(l->cache.bn_conv.delta);
            l->cache.bn_conv.delta = NULL;
        }
        if (l->cache.bn_conv.dL_dA != NULL) {
            tensor4D_free(l->cache.bn_conv.dL_dA);
            l->cache.bn_conv.dL_dA = NULL;
        }
        if (l->cache.bn_conv.dA_dZ != NULL) {
            tensor4D_free(l->cache.bn_conv.dA_dZ);
            l->cache.bn_conv.dA_dZ = NULL;
        }
        if (l->cache.bn_conv.x_normalized != NULL) {
            tensor4D_free(l->cache.bn_conv.x_normalized);
            l->cache.bn_conv.x_normalized = NULL;
        }
        if (l->cache.bn_conv.mean != NULL) {
            matrix_free(l->cache.bn_conv.mean);
            l->cache.bn_conv.mean = NULL;
        }
        if (l->cache.bn_conv.variance != NULL) {
            matrix_free(l->cache.bn_conv.variance);
            l->cache.bn_conv.variance = NULL;
        }
        if (l->cache.bn_conv.running_mean != NULL) {
            matrix_free(l->cache.bn_conv.running_mean);
            l->cache.bn_conv.running_mean = NULL;
        }
        if (l->cache.bn_conv.running_variance != NULL) {
            matrix_free(l->cache.bn_conv.running_variance);
            l->cache.bn_conv.running_variance = NULL;
        }
        if (l->cache.bn_conv.gamma != NULL) {
            matrix_free(l->cache.bn_conv.gamma);
            l->cache.bn_conv.gamma = NULL;
        }
        if (l->cache.bn_conv.beta != NULL) {
            matrix_free(l->cache.bn_conv.beta);
            l->cache.bn_conv.beta = NULL;
        }
        if (l->cache.bn_conv.gamma_grad != NULL) {
            matrix_free(l->cache.bn_conv.gamma_grad);
            l->cache.bn_conv.gamma_grad = NULL;
        }
        if (l->cache.bn_conv.beta_grad != NULL) {
            matrix_free(l->cache.bn_conv.beta_grad);
            l->cache.bn_conv.beta_grad = NULL;
        }
        break;
    }
    
    if (l->activation != NULL) {
        free(l->activation);
        l->activation = NULL;
    }

free(l);
}

int layer_get_n_units(Layer* l) {
    switch (l->l_type)
    {
    case INPUT:
    case DENSE:
    case OUTPUT:
        return l->params.dense.n_units;
        break;
    
    case CONV2D_INPUT:
    case CONV2D:
        return l->params.conv.n_units;
        break;

    case MAX_POOL:
        return l->params.max_pool.n_units;
        break;
    
    case FLATTEN:
        return l->params.flat.n_units;
        break;

    case BATCH_NORM_CONV2D:
        return l->params.bn_conv.n_units;
        break;
    }

    exit(1);
}

Matrix* layer_get_output_matrix(Layer* l) {
    switch (l->l_type)
    {
    case INPUT:
    case DENSE:
    case OUTPUT:
        return l->cache.dense.output;
        break;

    case FLATTEN:
        return l->cache.flat.output;    
        break;

    default:
        printf("Expected to return matrix, not tensor.");
        exit(1);
    }
}

Tensor4D* layer_get_output_tensor4D(Layer* l) {
    switch (l->l_type)
    {
    case CONV2D_INPUT:
    case CONV2D:
        return l->cache.conv.output;
        break;

    case MAX_POOL:
        return l->cache.max_pool.output;
        break;
    
    case BATCH_NORM_CONV2D:
        return l->cache.bn_conv.output;
        break;
    
    default:
        printf("Expected to return tensor, not matrix.");
        exit(1);
    }
}

Matrix* layer_get_delta_matrix(Layer* l) {
    switch (l->l_type)
    {
    case DENSE:
    case OUTPUT:
        return l->cache.dense.delta;
        break;
    
    case FLATTEN:
        return l->cache.flat.delta;
        break;
    
    default:
        printf("Object doesn't have delta matrix.");
        exit(1);
    }
}
Tensor4D* layer_get_delta_tensor4D(Layer* l) {
    switch (l->l_type)
    {
    case CONV2D:
        return l->cache.conv.delta;
        break;

    case MAX_POOL:
        return l->cache.max_pool.delta;
        break;
    
    default:
        printf("Object doesn't have delta tensor.");
        exit(1);
    }
}

unsigned int layer_get_sizeof_mem_allocated(Layer* l) {
    unsigned int size = 0;
    switch(l->l_type)
    {
        case OUTPUT:
            size = layer_output_get_sizeof_mem_allocated(l);
            break;
        
        case DENSE:
            size = layer_dense_get_sizeof_mem_allocated(l);
            break;
        
        case CONV2D:
            size = layer_conv2D_get_sizeof_mem_allocated(l);
            break;
        
        case FLATTEN:
            size = layer_flatten_get_sizeof_mem_allocated(l);
            break;

        case MAX_POOL:
            size = layer_max_pool_get_sizeof_mem_allocated(l);
            break;

        case BATCH_NORM_CONV2D:
            size = layer_batch_norm_conv2D_get_sizeof_mem_allocated(l);
            break;
    }

    return size;
}

void layer_dense_compile(Layer* l, ActivationType act_type, int act_param, int batch_size) {
    l->activation = activation_new(
        act_type,
        act_param
    );
    l->cache.dense.weight = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias = matrix_new(
        1,
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
    l->cache.dense.weight_grad = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias_grad = matrix_new(
        1,
        layer_get_n_units(l)
    );
    l->cache.dense.dL_dA = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dA_dZ = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );

    matrix_zero(l->cache.dense.bias);

    switch (act_type)
    {
    case SIGMOID:
        #ifdef SINGLE_PRECISION

        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            (nn_float)0.0,
            sqrtf((nn_float)2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer)))
        );

        #endif
        #ifdef DOUBLE_PRECISION

        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            (nn_float)0.0,
            sqrt((nn_float)2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer)))
        );        

        #endif
        break;
    
    case RELU:
    case LRELU:
    case ELU:
        #ifdef SINGLE_PRECISION

        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            (nn_float)0.0,
            sqrtf((nn_float)2.0/layer_get_n_units(l->prev_layer))
        );

        #endif
        #ifdef DOUBLE_PRECISION

        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            (nn_float)0.0,
            sqrt((nn_float)2.0/layer_get_n_units(l->prev_layer))
        );        
        
        #endif
        break;
    
    default:
        printf("Unknown activation type.");
        exit(1);
        break;
    }
}

void layer_output_compile(Layer* l, Loss* loss, int batch_size) {
    if (layer_get_n_units(l) == 1)
        l->activation = activation_new(SIGMOID, (nn_float)0.0);
    if (layer_get_n_units(l) > 1)
        l->activation = activation_new(SOFTMAX, (nn_float)0.0);
    l->activation->y_true_batch = batch_matrix_new(
        batch_size,
        layer_get_n_units(l)
    );

    loss->loss_m = matrix_new(batch_size, layer_get_n_units(l));

    l->cache.dense.weight = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias = matrix_new(
        1,
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
    l->cache.dense.weight_grad = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias_grad = matrix_new(
        1,
        layer_get_n_units(l)
    );
    l->cache.dense.dL_dA = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.dense.dA_dZ = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );

    matrix_zero(l->cache.dense.bias);
    switch (l->activation->type)
    {
    case SIGMOID:
    case SOFTMAX:
        #ifdef SINGLE_PRECISION

        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            (nn_float)0.0,
            sqrtf((nn_float)2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer)))
        );

        #endif
        #ifdef DOUBLE_PRECISION

        matrix_fill_normal_distribution(
            l->cache.dense.weight,
            (nn_float)0.0,
            sqrt((nn_float)2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer)))
        ); 

        #endif
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

    l->params.conv.n_filter_channels = input_channels;

    if (l->next_layer->l_type == BATCH_NORM_CONV2D) {
        l->activation = activation_new(
            IDENTITY,
            0.0
        );
    }
    else {
        l->activation = activation_new(
            act_type,
            act_param
        );
    }
    l->cache.conv.weight = tensor4D_new(
        l->params.conv.filter_size,
        l->params.conv.filter_size,
        l->params.conv.n_filter_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.weight_flip = tensor4D_new(
    l->params.conv.filter_size,
    l->params.conv.filter_size,
    l->params.conv.n_filter_channels,
    l->params.conv.n_filters
    );
    l->cache.conv.bias = matrix_new(
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
    l->cache.conv.weight_grad = tensor4D_new(
        l->params.conv.filter_size,
        l->params.conv.filter_size,
        l->params.conv.n_filter_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.bias_grad = matrix_new(
        1,
        l->params.conv.n_filters
    );
    l->cache.conv.dL_dA = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        batch_size
    );
    l->cache.conv.dA_dZ = tensor4D_new(
        output_height,
        output_width,
        l->params.conv.n_filters,
        batch_size
    );

    int pad_h = output_height + 2*(filter_height-1);
    int pad_w = output_width + 2*(filter_width-1);
    l->cache.conv.padding = tensor4D_new(
        pad_h,
        pad_w,
        l->params.conv.n_filters,
        batch_size
    );


    l->cache.conv.fp_im2col_input = tensor3D_new(
        output_height * output_width,
        filter_height * filter_width * l->params.conv.n_filter_channels,
        batch_size
    );
    l->cache.conv.fp_im2col_kernel = matrix_new(
        filter_height * filter_width * l->params.conv.n_filter_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.fp_im2col_output = tensor3D_new(
        l->params.conv.n_filters,
        output_height * output_width,
        batch_size
    );
    l->cache.conv.dL_dW_im2col_input = tensor3D_new(
        output_height * output_width,
        filter_height * filter_width * l->params.conv.n_filter_channels,
        batch_size
    );
    l->cache.conv.dL_dW_im2col_kernel = tensor3D_new(
        output_height * output_width,
        l->params.conv.n_filters,
        batch_size
    );
    l->cache.conv.dL_dW_im2col_output = tensor3D_new(
        l->params.conv.n_filters,
        filter_height * filter_width * l->params.conv.n_filter_channels,
        batch_size
    );
    l->cache.conv.dL_dW_im2col_output_sum = matrix_new(
        l->params.conv.n_filters,
        filter_height * filter_width * l->params.conv.n_filter_channels
    );
    l->cache.conv.delta_im2col_input = tensor3D_new(
        input_height * input_width,
        filter_height * filter_width * l->params.conv.n_filters,
        batch_size
    );
    l->cache.conv.delta_im2col_kernel = matrix_new(
        filter_height * filter_width * l->params.conv.n_filters,
        l->params.conv.n_filter_channels
    );
    l->cache.conv.delta_im2col_output = tensor3D_new(
        l->params.conv.n_filter_channels,
        input_height * input_width,
        batch_size
    );

    l->params.conv.n_units = 
        l->cache.conv.output->n_rows *
        l->cache.conv.output->n_cols *
        l->cache.conv.output->n_channels;
    
    matrix_zero(l->cache.conv.bias);

    switch (l->activation->type)
    {
    case SIGMOID:
        #ifdef SINGLE_PRECISION

        tensor4D_fill_normal_distribution(
            l->cache.conv.weight,
            (nn_float)0.0,
            sqrtf((nn_float)2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer)))
        );

        #endif
        #ifdef DOUBLE_PRECISION

        tensor4D_fill_normal_distribution(
            l->cache.conv.weight,
            (nn_float)0.0,
            sqrt((nn_float)2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer)))
        );
        
        #endif
        break;
    
    case RELU:
    case LRELU:
    case ELU:
    case IDENTITY:
        #ifdef SINGLE_PRECISION

        tensor4D_fill_normal_distribution(
            l->cache.conv.weight,
            (nn_float)0.0,
            sqrtf((nn_float)2.0/layer_get_n_units(l->prev_layer))
        );

        #endif
        #ifdef DOUBLE_PRECISION

        tensor4D_fill_normal_distribution(
            l->cache.conv.weight,
            (nn_float)0.0,
            sqrt((nn_float)2.0/layer_get_n_units(l->prev_layer))
        );
        
        #endif
        break;
    
    default:
        printf("Unknown activation type.");
        exit(1);
        break;
    }
    
    #ifdef IM2COL_CONV
    kernel_into_im2col_fwise(
        l->cache.conv.weight,
        false,
        l->cache.conv.fp_im2col_kernel
    );
    #endif
}

void layer_flatten_compile(Layer* l, int batch_size) {
    int input_channels = layer_get_output_tensor4D(l->prev_layer)->n_channels;
    int input_height = layer_get_output_tensor4D(l->prev_layer)->n_rows;
    int input_width = layer_get_output_tensor4D(l->prev_layer)->n_cols;
    l->params.flat.n_units = input_channels * input_height * input_width;
    l->cache.flat.output = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
    l->cache.flat.delta = matrix_new(
        batch_size,
        layer_get_n_units(l)
    );
}

void layer_max_pool_compile(Layer* l, int batch_size) {
    int input_height = layer_get_output_tensor4D(l->prev_layer)->n_rows;
    int input_width = layer_get_output_tensor4D(l->prev_layer)->n_cols;
    int input_channels = layer_get_output_tensor4D(l->prev_layer)->n_channels;
    int filter_height = l->params.max_pool.filter_size;
    int filter_width = l->params.max_pool.filter_size;
    int stride = l->params.max_pool.stride;
    int output_height = floor((input_height - filter_height) / stride) + 1;
    int output_width = floor((input_width - filter_width) / stride) + 1;

    l->cache.max_pool.output = tensor4D_new(
        output_height,
        output_width,
        input_channels,
        batch_size
    );
    l->cache.max_pool.delta = tensor4D_new(
        output_height,
        output_width,
        input_channels,
        batch_size
    );
    l->cache.max_pool.argmax = tensor4D_uint16_new(
        output_height,
        output_width,
        input_channels,
        batch_size
    );

    l->params.max_pool.n_units = 
        l->cache.max_pool.output->n_rows *
        l->cache.max_pool.output->n_cols *
        l->cache.max_pool.output->n_channels;
}

void layer_batch_norm_conv2D_compile(Layer* l, ActivationType act_type, int act_param, int batch_size) {
    int input_height = layer_get_output_tensor4D(l->prev_layer)->n_rows;
    int input_width = layer_get_output_tensor4D(l->prev_layer)->n_cols;
    int input_channels = layer_get_output_tensor4D(l->prev_layer)->n_channels;
    int output_height = input_height;
    int output_width = input_width;
    int output_channels = input_channels;
    int output_filters = batch_size;

    l->params.bn_conv.output_height = output_height;
    l->params.bn_conv.output_width = output_width;
    l->params.bn_conv.output_channels = output_channels;
    l->params.bn_conv.output_filters = output_filters;
    l->params.bn_conv.n_units = output_height * output_width * output_channels;


    if (l->prev_layer->l_type == CONV2D) {
        l->activation = activation_new(
            act_type,
            act_param
        );
    }
    else {
        l->activation = activation_new(
            IDENTITY,
            0.0
        );
    }

    l->cache.bn_conv.output = tensor4D_new(
        output_height,
        output_width,
        output_channels,
        output_filters
    );
    l->cache.bn_conv.z = tensor4D_new(
        output_height,
        output_width,
        output_channels,
        output_filters
    );
    l->cache.bn_conv.delta = tensor4D_new(
        output_height,
        output_width,
        output_channels,
        output_filters
    );
    l->cache.bn_conv.dL_dA = tensor4D_new(
        output_height,
        output_width,
        output_channels,
        output_filters
    );
    l->cache.bn_conv.dA_dZ = tensor4D_new(
        output_height,
        output_width,
        output_channels,
        output_filters
    );
    l->cache.bn_conv.x_normalized = tensor4D_new(
        input_height,
        input_width,
        input_channels,
        batch_size
    );
    
    l->cache.bn_conv.mean = matrix_new(
        1,
        input_channels
    );
    l->cache.bn_conv.variance = matrix_new(
        1,
        input_channels
    );
    l->cache.bn_conv.running_mean = matrix_new(
        1,
        input_channels
    );
    l->cache.bn_conv.running_variance = matrix_new(
        1,
        input_channels
    );
    l->cache.bn_conv.gamma = matrix_new(
        1,
        input_channels
    );
    l->cache.bn_conv.beta = matrix_new(
        1,
        input_channels
    );
    l->cache.bn_conv.gamma_grad = matrix_new(
        1,
        input_channels
    );
    l->cache.bn_conv.beta_grad = matrix_new(
        1,
        input_channels
    );

    matrix_fill(l->cache.bn_conv.gamma, 1.0f);
    matrix_zero(l->cache.bn_conv.beta);
    matrix_fill(l->cache.bn_conv.running_variance, 1.0f);
}

void layer_input_fp(Layer* l, Batch* train_batch) {
    l->cache.dense.output = train_batch->data.matrix;
}

void layer_conv2D_input_fp(Layer* l, Batch* train_batch) {
    l->cache.conv.output = train_batch->data.tensor;
}

void layer_dense_fp(Layer* l) {
    matrix_dot_into(layer_get_output_matrix(
        l->prev_layer), 
        l->cache.dense.weight, 
        l->cache.dense.z,
        false,
        false
    );
    bias_add_to_dense_z(l->cache.dense.bias, l->cache.dense.z);
    apply_activation_func_into(l->activation, l->cache.dense.z, l->cache.dense.output);
}

void layer_output_fp(Layer* l, Batch* label_batch) {
    matrix_dot_into(layer_get_output_matrix(
        l->prev_layer), 
        l->cache.dense.weight, 
        l->cache.dense.z,
        false,
        false
    );
    bias_add_to_dense_z(l->cache.dense.bias, l->cache.dense.z);
    l->activation->y_true_batch = label_batch;
    apply_activation_func_into(l->activation, l->cache.dense.z, l->cache.dense.output);
}

void layer_conv2D_fp(Layer* l) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* weight = l->cache.conv.weight;
    Tensor4D* z = l->cache.conv.z;
    Matrix* bias = l->cache.conv.bias;
    Tensor4D* output = l->cache.conv.output;

    #ifdef IM2COL_CONV
    
    #pragma omp parallel for schedule(static)
    for (int n=0; n<input->n_filters; n++) {
        Matrix* im2col_input = l->cache.conv.fp_im2col_input->channels[n];
        Matrix* im2col_output = l->cache.conv.fp_im2col_output->channels[n];
        int out_h = z->filters[n]->n_rows;
        int out_w = z->filters[n]->n_cols;
        int out_size = out_h * out_w;

        input_into_im2col_fwise(
            input,
            n,
            weight,
            l->params.conv.stride,
            0,
            im2col_input
        );
        matrix_dot_into(
            l->cache.conv.fp_im2col_kernel,
            im2col_input,
            im2col_output,
            true,
            true
        );

        for (int i=0; i<weight->n_filters; i++) {
            nn_float* src = im2col_output->entries + i*out_size;
            nn_float* dst = z->filters[n]->channels[i]->entries;
            memcpy(dst, src, out_size*sizeof(nn_float));
        }
        for (int i=0; i<weight->n_filters; i++) {
            matrix_add_scalar_inplace(
                matrix_get(bias, 0, i),
                z->filters[n]->channels[i]
            );
            apply_activation_func_into(
                l->activation,
                z->filters[n]->channels[i],
                output->filters[n]->channels[i]
            );
        }
    }

    #else

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n=0; n<input->n_filters; n++) {
        for (int i=0; i<weight->n_filters; i++) {
            matrix_zero(z->filters[n]->channels[i]);
            tensor3D_acc_correlate_into(
                input->filters[n],
                weight->filters[i],
                z->filters[n]->channels[i],
                l->params.conv.stride,
                VALID
            );
            matrix_add_scalar_inplace(
                matrix_get(bias, 0, i),
                z->filters[n]->channels[i]
            );
            apply_activation_func_into(
                l->activation,
                z->filters[n]->channels[i],
                output->filters[n]->channels[i]
            );
        }
    }

    #endif
}

void layer_flatten_fp(Layer* l) {
    Tensor4D* t = layer_get_output_tensor4D(l->prev_layer);
    Matrix* m = l->cache.flat.output;
    tensor4D_into_matrix_fwise(
        t,
        m,
        false,
        false
    );
}

void layer_max_pool_fp(Layer* l) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* output = layer_get_output_tensor4D(l);
    Tensor4D_uint16* argmax = l->cache.max_pool.argmax;

    const int filter_size = l->params.max_pool.filter_size;
    const int stride = l->params.max_pool.stride;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n=0; n<input->n_filters; n++) {
        for (int c=0; c<input->n_channels; c++) {
            matrix_max_pool_into(
                input->filters[n]->channels[c],
                output->filters[n]->channels[c],
                argmax->filters[n]->channels[c],
                filter_size,
                stride
            );
        }
    }
}

void layer_batch_norm_conv2D_fp(Layer* l, bool training) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* input_normalized = l->cache.bn_conv.x_normalized;
    Tensor4D* z = l->cache.bn_conv.z;
    Tensor4D* output = l->cache.bn_conv.output;
    nn_float momentum = l->params.bn_conv.momentum;

    if (training) {
        // per-channel mean calculations
        #pragma omp parallel for schedule(static)
        for (int c=0; c<input->n_channels; c++) {
            nn_float sum = (nn_float)0.0;

            for (int n=0; n<input->n_filters; n++) {
                sum += matrix_sum(input->filters[n]->channels[c]);
            }

            nn_float mean = sum / (input->n_filters * input->n_rows * input->n_cols);
            matrix_assign(l->cache.bn_conv.mean, 0, c, mean);

            // running mean calculations
            nn_float running_mean = matrix_get(l->cache.bn_conv.running_mean, 0, c);
            running_mean = momentum * mean + ((nn_float)1.0 - momentum) * running_mean;
            matrix_assign(l->cache.bn_conv.running_mean, 0, c, running_mean);
        }

        // per-channel variance calculations
        #pragma omp parallel for schedule(static)
        for (int c=0; c<input->n_channels; c++) {
            nn_float mean = matrix_get(l->cache.bn_conv.mean, 0, c);
            nn_float sum = (nn_float)0.0;

            for (int n=0; n<input->n_filters; n++) {
                nn_float* channel = input->filters[n]->channels[c]->entries;

                for (int i=0; i<input->n_rows * input->n_cols; i++) {
                    nn_float var = channel[i] - mean;
                    sum += var * var;
                }
            }

            nn_float channel_var = sum / (input->n_filters * input->n_rows * input->n_cols);
            matrix_assign(l->cache.bn_conv.variance, 0, c, channel_var);

            // running variance calculations
            nn_float running_var = matrix_get(l->cache.bn_conv.running_variance, 0, c);
            running_var = momentum * channel_var + ((nn_float)1.0 - momentum) * running_var;
            matrix_assign(l->cache.bn_conv.running_variance, 0, c, running_var);
        }
    }
    
    // normalization and scaling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n=0; n<input->n_filters; n++) {
        for (int c=0; c<input->n_channels; c++) {
            nn_float mean, var;
            if (training) {
                mean = matrix_get(l->cache.bn_conv.mean, 0, c);
                var = matrix_get(l->cache.bn_conv.variance, 0, c);
            }
            else {
                mean = matrix_get(l->cache.bn_conv.running_mean, 0, c);
                var = matrix_get(l->cache.bn_conv.running_variance, 0, c);
            }

            nn_float gamma = matrix_get(l->cache.bn_conv.gamma, 0, c);
            nn_float beta = matrix_get(l->cache.bn_conv.beta, 0, c);
            nn_float inv_std;

            #ifdef SINGLE_PRECISION
            inv_std = (nn_float)1.0 / sqrtf(var + 1e-5);
            #endif
            #ifdef DOUBLE_PRECISION
            inv_std = (nn_float)1.0 / sqrt(var + 1e-5);
            #endif

            nn_float* input_channel = input->filters[n]->channels[c]->entries;
            nn_float* normalized_channel = input_normalized->filters[n]->channels[c]->entries;
            nn_float* z_channel = z->filters[n]->channels[c]->entries;

            for (int i=0; i<input->n_rows * input->n_cols; i++) {
                nn_float x = input_channel[i];
                nn_float x_normalized = (x - mean) * inv_std;
                normalized_channel[i] = x_normalized;
                z_channel[i] = gamma * x_normalized + beta;
            }

            apply_activation_func_into(
                l->activation,
                z->filters[n]->channels[c],
                output->filters[n]->channels[c]
            );
        }
    }
}

void layer_output_bp(Layer* l, Loss* loss, Batch* label_batch) {
    // dL_dZ calculation
    apply_loss_dA_into(
        loss, 
        l->cache.dense.output, 
        label_batch->data.matrix, 
        l->cache.dense.dL_dA
    );

    // dL_dZ calculation
    apply_activation_dZ_into(
        l->activation, 
        l->cache.dense.z, 
        l->cache.dense.dA_dZ
    );
    matrix_multiply_into(
        l->cache.dense.dL_dA, 
        l->cache.dense.dA_dZ, 
        l->cache.dense.delta
    );

    // dL_dW calculation
    matrix_dot_into(
        layer_get_output_matrix(l->prev_layer), 
        l->cache.dense.delta, 
        l->cache.dense.weight_grad,
        true,
        false
    );

    // dL_dB calculation
    matrix_sum_axis_into(
        l->cache.dense.delta, 
        1, 
        l->cache.dense.bias_grad
    );
}

void layer_dense_bp(Layer* l) {
    // dL_dA calculation
    if (l->next_layer->l_type == DENSE || l->next_layer->l_type == OUTPUT) {
        bp_delta_from_dense(l->next_layer, l->cache.dense.dL_dA);
    }

    // dL_dZ calculation
    apply_activation_dZ_into(
        l->activation, 
        l->cache.dense.z, 
        l->cache.dense.dA_dZ
    );
    matrix_multiply_into(
        l->cache.dense.dL_dA, 
        l->cache.dense.dA_dZ, 
        l->cache.dense.delta
    );

    // dL_dW calculation
    matrix_dot_into(
        layer_get_output_matrix(l->prev_layer), 
        l->cache.dense.delta, 
        l->cache.dense.weight_grad,
        true,
        false
    );

    // dL_dB calculation
    matrix_sum_axis_into(
        l->cache.dense.delta, 
        1, 
        l->cache.dense.bias_grad
    );
}

void layer_conv2D_bp(Layer* l) {
    Tensor4D* weight = l->cache.conv.weight;
    Tensor4D* weight_grad = l->cache.conv.weight_grad;
    Matrix* bias_grad = l->cache.conv.bias_grad;
    Tensor4D* z = l->cache.conv.z;
    Tensor4D* dA_dZ = l->cache.conv.dA_dZ;
    Tensor4D* dL_dA = l->cache.conv.dL_dA;
    Tensor4D* delta = l->cache.conv.delta;

    // dL_dA calculation
    if (l->next_layer->l_type == FLATTEN) {
        bp_delta_from_flatten(l->next_layer, dL_dA);
    }
    else if (l->next_layer->l_type == MAX_POOL) {
        bp_delta_from_max_pool(l->next_layer, dL_dA);
    }
    else if (l->next_layer->l_type == CONV2D) {
        bp_delta_from_conv2D(l->next_layer, dL_dA);
    }
    else if (l->next_layer->l_type == BATCH_NORM_CONV2D) {
        bp_delta_from_batch_norm_conv2D(l->next_layer, dL_dA);
    }

    // dL_dZ calculation
    for (int n=0; n<delta->n_filters; n++) {
        for (int c=0; c<delta->n_channels; c++) {
            apply_activation_dZ_into(
                l->activation,
                z->filters[n]->channels[c],
                dA_dZ->filters[n]->channels[c]
            );
            matrix_multiply_into(
                dL_dA->filters[n]->channels[c],
                dA_dZ->filters[n]->channels[c],
                delta->filters[n]->channels[c]
            );
        }
    }

    // dL_dW calculation
    {
        Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
        
        #ifdef IM2COL_CONV

        Matrix* output_sum_mat = l->cache.conv.dL_dW_im2col_output_sum;
        matrix_zero(output_sum_mat);

        #pragma omp parallel for schedule(static)
        for (int n=0; n<input->n_filters; n++) {
            Matrix* input_mat = l->cache.conv.dL_dW_im2col_input->channels[n];
            Matrix* kernel_mat = l->cache.conv.dL_dW_im2col_kernel->channels[n];
            Matrix* output_mat = l->cache.conv.dL_dW_im2col_output->channels[n];

            input_into_im2col_fwise(
                input,
                n,
                weight,
                l->params.conv.stride,
                0,
                input_mat
            );
            delta_into_im2col_fwise(
                delta,
                n,
                kernel_mat
            );
            matrix_dot_into(
                kernel_mat,
                input_mat,
                output_mat,
                true,
                false
            );
        }

        tensor3D_sum_element_wise_into(
            l->cache.conv.dL_dW_im2col_output,
            output_sum_mat
        );
        
        int k = weight->n_rows * weight->n_cols * weight->n_channels;
        for (int f=0; f<weight->n_filters; f++) {
            nn_float* row = output_sum_mat->entries + f * k;
            for (int c=0; c<weight->n_channels; c++) {
                nn_float* src = row + c * weight->n_rows * weight->n_cols;
                nn_float* dst = weight_grad->filters[f]->channels[c]->entries;
                memcpy(dst, src, weight->n_rows * weight->n_cols * sizeof(nn_float));
            }
        }

        #else

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i=0; i<weight->n_filters; i++) {
            for (int j=0; j<weight->n_channels; j++) {
                matrix_zero(weight_grad->filters[i]->channels[j]);
                for (int n=0; n<delta->n_filters; n++) {
                    matrix_acc_correlate_into(
                        input->filters[n]->channels[j],
                        delta->filters[n]->channels[i],
                        weight_grad->filters[i]->channels[j],
                        l->params.conv.stride,
                        VALID
                    );
                }
            }
        }

        #endif
    }

    // dL_dB calculation
    for (int i=0; i<weight->n_filters; i++) {
        nn_float sum = (nn_float)0.0;
        for (int n=0; n<delta->n_filters; n++) {
            sum += matrix_sum(delta->filters[n]->channels[i]);
        }
        matrix_assign(bias_grad, 0, i, sum);
    }
}

void layer_max_pool_bp(Layer* l) {
    // dL_dZ calculation
    if (l->next_layer->l_type == CONV2D) {
        bp_delta_from_conv2D(l->next_layer, l->cache.max_pool.delta);
    }
    else if (l->next_layer->l_type == FLATTEN) {
        bp_delta_from_flatten(l->next_layer, l->cache.max_pool.delta);     
    }
    else if (l->next_layer->l_type == BATCH_NORM_CONV2D) {
        bp_delta_from_batch_norm_conv2D(l->next_layer, l->cache.max_pool.delta);
    }
}

void layer_flatten_bp(Layer* l) {
    // dL_dZ calculation
    if (l->next_layer->l_type == DENSE || l->next_layer->l_type == OUTPUT) {
        bp_delta_from_dense(l->next_layer, l->cache.flat.delta);
    }
}

void layer_batch_norm_conv2D_bp(Layer* l) {
    Tensor4D* delta = l->cache.bn_conv.delta;
    Tensor4D* dL_dA = l->cache.bn_conv.dL_dA;
    Tensor4D* dA_dZ = l->cache.bn_conv.dA_dZ;
    Tensor4D* z = l->cache.bn_conv.z;

    // dL_dA calculation
    if (l->next_layer->l_type == CONV2D) {
        bp_delta_from_conv2D(l->next_layer, dL_dA);
    }
    else if (l->next_layer->l_type == FLATTEN) {
        bp_delta_from_flatten(l->next_layer, dL_dA);
    }

    // dL_dZ calculation
    for (int n=0; n<delta->n_filters; n++) {
        for (int c=0; c<delta->n_channels; c++) {
            apply_activation_dZ_into(
                l->activation,
                z->filters[n]->channels[c],
                dA_dZ->filters[n]->channels[c]
            );
            matrix_multiply_into(
                dL_dA->filters[n]->channels[c],
                dA_dZ->filters[n]->channels[c],
                delta->filters[n]->channels[c]
            );
        }
    }

    // dL_dgamma and dL_dbeta
    Matrix* beta_grad = l->cache.bn_conv.beta_grad;
    Matrix* gamma_grad = l->cache.bn_conv.gamma_grad;
    Tensor4D* x_norm = l->cache.bn_conv.x_normalized;

    matrix_zero(beta_grad);
    matrix_zero(gamma_grad);

    for (int c=0; c<delta->n_channels; c++) {
        nn_float b_sum = (nn_float)0.0;
        nn_float g_sum = (nn_float)0.0;
        for (int n=0; n<delta->n_filters; n++) {
            nn_float* delta_row = delta->filters[n]->channels[c]->entries;
            nn_float* x_norm_row = x_norm->filters[n]->channels[c]->entries;
            for (int i=0; i<delta->n_rows * delta->n_cols; i++) {
                b_sum += delta_row[i];
                g_sum += delta_row[i] * x_norm_row[i];
            }
        }

        matrix_assign(beta_grad, 0, c, b_sum);
        matrix_assign(gamma_grad, 0, c, g_sum);
    }
}

void bp_delta_from_dense(Layer* from, Matrix* to) {
    matrix_dot_into(
        from->cache.dense.delta,
        from->cache.dense.weight,
        to,
        false,
        true
    );
}

void bp_delta_from_conv2D(Layer* from, Tensor4D* to) {
    Tensor4D* dL_dA = to;
    Tensor4D* delta_next = from->cache.conv.delta;
    Tensor4D* filter_next = from->cache.conv.weight;
    Tensor4D* filter_flip_next = from->cache.conv.weight_flip;
    Tensor4D* padding_next = from->cache.conv.padding;

    #ifdef IM2COL_CONV

    kernel_into_im2col_chwise(
        filter_next,
        true,
        from->cache.conv.delta_im2col_kernel
    );

    #pragma omp parallel for schedule(static)
    for (int n=0; n<dL_dA->n_filters; n++) {
        input_into_im2col_fwise(
            delta_next,
            n,
            filter_next,
            1,
            filter_next->n_rows - 1,
            from->cache.conv.delta_im2col_input->channels[n]
        );
        matrix_dot_into(
            from->cache.conv.delta_im2col_kernel,
            from->cache.conv.delta_im2col_input->channels[n],
            from->cache.conv.delta_im2col_output->channels[n],
            true,
            true
        );

        Matrix* output_im2col_mat = from->cache.conv.delta_im2col_output->channels[n];
        for (int c=0; c<dL_dA->n_channels; c++) {
            nn_float* src = output_im2col_mat->entries + c * dL_dA->n_rows * dL_dA->n_cols;
            nn_float* dst = dL_dA->filters[n]->channels[c]->entries;
            memcpy(dst, src, dL_dA->n_rows * dL_dA->n_cols * sizeof(nn_float));
        }
    }

    #else

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n=0; n<dL_dA->n_filters; n++) {
        for (int i=0; i<dL_dA->n_channels; i++) {
            matrix_zero(dL_dA->filters[n]->channels[i]);
            for (int f=0; f<filter_next->n_filters; f++) {
                matrix_acc_convolve_full_into(
                    delta_next->filters[n]->channels[f],
                    filter_flip_next->filters[f]->channels[i],
                    dL_dA->filters[n]->channels[i],
                    padding_next->filters[n]->channels[f]
                );
            }
        }
    }

    #endif
}

void bp_delta_from_max_pool(Layer* from, Tensor4D* to) {
    Tensor4D* dL_dA = to;
    Tensor4D* delta_next = layer_get_delta_tensor4D(from);
    Tensor4D* output_next = layer_get_output_tensor4D(from);
    Tensor4D_uint16* argmax = from->cache.max_pool.argmax;
    int filter_size = from->params.max_pool.filter_size;
    int stride = from->params.max_pool.stride;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n=0; n<output_next->n_filters; n++) {
        for (int c=0; c<output_next->n_channels; c++) {
            Matrix* dL_dA_mat = dL_dA->filters[n]->channels[c];
            matrix_zero(dL_dA_mat);
            Matrix* delta_next_mat = delta_next->filters[n]->channels[c];
            Matrix_uint16* argmax_mat = argmax->filters[n]->channels[c];
            int out_h = to->n_rows;
            int out_w = to->n_cols;
            int out_h_next = output_next->n_rows;
            int out_w_next = output_next->n_cols;
            for (int i=0; i<out_h_next; i++) {
                for (int j=0; j<out_w_next; j++) {
                    int grad_idx = argmax_mat->entries[i*out_w_next + j];
                    int ker_h_offset = grad_idx / filter_size;
                    int ker_w_offset = grad_idx % filter_size;
                    int in_row = i*stride + ker_h_offset;
                    int in_col = j*stride + ker_w_offset;
                    dL_dA_mat->entries[in_row*out_w + in_col] +=
                        delta_next_mat->entries[i*out_w_next + j];
                }
            }

        }
    }
}

void bp_delta_from_flatten(Layer* from, Tensor4D* to) {
    matrix_into_tensor4D(
        from->cache.flat.delta,
        to
    );
}

void bp_delta_from_batch_norm_conv2D(Layer* from, Tensor4D* to) {
    Tensor4D* bn_input = layer_get_output_tensor4D(from->prev_layer);

    #pragma omp parallel for schedule(static)
    for (int c=0; c<to->n_channels; c++) {
        nn_float mean_c = matrix_get(from->cache.bn_conv.mean, 0, c);
        nn_float var_c = matrix_get(from->cache.bn_conv.variance, 0, c);
        nn_float gamma_c = matrix_get(from->cache.bn_conv.gamma, 0, c);
        nn_float dvar_c = (nn_float)0.0;
        nn_float dmean_c = (nn_float)0.0;
        int m = to->n_filters * to->n_rows * to->n_cols;

        #ifdef SINGLE_PRECISION
        nn_float inv_std = (nn_float)1.0 / sqrtf(var_c + 1e-5);
        nn_float inv_pow_3_2 = (nn_float)1.0 / (sqrtf(var_c + 1e-5) * (var_c + 1e-5));
        #endif
        #ifdef DOUBLE_PRECISION
        nn_float inv_std = (nn_float)1.0 / sqrt(var_c + 1e-5);
        nn_float inv_pow_3_2 = (nn_float)1.0 / (sqrt(var_c + 1e-5) * (var_c + 1e-5));        
        #endif

        for (int n=0; n<to->n_filters; n++) {
            nn_float* delta = from->cache.bn_conv.delta->filters[n]->channels[c]->entries;
            nn_float* x = bn_input->filters[n]->channels[c]->entries;

            for (int i=0; i<to->n_rows * to->n_cols; i++) {
                // dL_dvar
                dvar_c += delta[i] * gamma_c * (x[i]-mean_c) * inv_pow_3_2 * (nn_float)-0.5;

                // dL_dmean (part 1/2)
                dmean_c += delta[i] * gamma_c * (-inv_std);
            }
        }

        nn_float sum = (nn_float)0.0;
        for (int n=0; n<to->n_filters; n++) {
            nn_float* x = bn_input->filters[n]->channels[c]->entries;

            for (int i=0; i<to->n_rows * to->n_cols; i++) {
                sum += x[i] - mean_c;
            }
        }

        // dL_dmean (part 2/2)
        dmean_c += dvar_c * sum / (nn_float)m *(nn_float)-2.0;

        for (int n=0; n<to->n_filters; n++) {
            nn_float* delta = from->cache.bn_conv.delta->filters[n]->channels[c]->entries;
            nn_float* x = bn_input->filters[n]->channels[c]->entries;
            nn_float* dL_dX = to->filters[n]->channels[c]->entries;
            for (int i=0; i<to->n_rows * to->n_cols; i++) {
                nn_float term1 = delta[i] * gamma_c * inv_std;
                nn_float term2 = dvar_c * (x[i] - mean_c) / (nn_float)m * (nn_float)2.0;
                nn_float term3 = dmean_c / (nn_float)m;

                // dL_dX
                dL_dX[i] = term1 + term2 + term3;
            }
        }
    }
}

void layer_dense_update_weights(Layer* l, Optimizer* opt) {
    opt->update_dense_weights(
        l->cache.dense.weight, 
        l->cache.dense.weight_grad, 
        opt,
        l->layer_idx
    );
    opt->update_dense_bias(
        l->cache.dense.bias, 
        l->cache.dense.bias_grad, 
        opt,
        l->layer_idx
    );
}

void layer_conv2D_update_weights(Layer* l, Optimizer* opt) {
    Tensor4D* weight = l->cache.conv.weight;
    Tensor4D* weight_flip = l->cache.conv.weight_flip;
    Matrix* bias = l->cache.conv.bias;
    Tensor4D* weight_grad = l->cache.conv.weight_grad;
    Matrix* bias_grad = l->cache.conv.bias_grad;
    opt->update_conv_weights(
        weight,
        weight_grad,
        opt,
        l->layer_idx
    );
    opt->update_conv_bias(
        bias,
        bias_grad,
        opt,
        l->layer_idx
    );

    tensor4D_flip_into(
        weight,
        weight_flip
    );

    #ifdef IM2COL_CONV
    kernel_into_im2col_fwise(weight, false, l->cache.conv.fp_im2col_kernel);
    #endif
}

void layer_batch_norm_conv2D_update_weights(Layer* l, Optimizer* opt) {
    opt->update_batch_norm_gamma(
        l->cache.bn_conv.gamma, 
        l->cache.bn_conv.gamma_grad, 
        opt,
        l->layer_idx
    );
    opt->update_batch_norm_beta(
        l->cache.bn_conv.beta, 
        l->cache.bn_conv.beta_grad, 
        opt,
        l->layer_idx
    );   
}

unsigned long layer_output_get_sizeof_mem_allocated(Layer* l) {
    unsigned long size = 0;
    size += sizeof(l);
    size += sizeof(l->params);
    size += sizeof(l->params.dense);
    size += sizeof(l->cache.dense);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.output);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.z);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.weight);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.bias);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.delta);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.weight_grad);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.bias_grad);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.dL_dA);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.dA_dZ);

    return size;
}

unsigned long layer_dense_get_sizeof_mem_allocated(Layer* l) {
    unsigned long size = 0;
    size += sizeof(l);
    size += sizeof(l->params);
    size += sizeof(l->params.dense);
    size += sizeof(l->cache.dense);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.output);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.z);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.weight);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.bias);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.delta);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.weight_grad);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.bias_grad);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.dL_dA);
    size += matrix_get_sizeof_mem_allocated(l->cache.dense.dA_dZ);

    return size;
}

unsigned long layer_conv2D_get_sizeof_mem_allocated(Layer* l) {
    unsigned long size = 0;
    size += sizeof(l);
    size += sizeof(l->params);
    size += sizeof(l->params.conv);
    size += sizeof(l->cache.conv);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.conv.output);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.conv.z);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.conv.weight);
    size += matrix_get_sizeof_mem_allocated(l->cache.conv.bias);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.conv.delta);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.conv.weight_grad);
    size += matrix_get_sizeof_mem_allocated(l->cache.conv.bias_grad);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.conv.dL_dA);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.conv.dA_dZ);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.conv.padding);
    size += tensor3D_get_sizeof_mem_allocated(l->cache.conv.fp_im2col_input);
    size += matrix_get_sizeof_mem_allocated(l->cache.conv.fp_im2col_kernel);
    size += tensor3D_get_sizeof_mem_allocated(l->cache.conv.fp_im2col_output);
    size += tensor3D_get_sizeof_mem_allocated(l->cache.conv.dL_dW_im2col_input);
    size += tensor3D_get_sizeof_mem_allocated(l->cache.conv.dL_dW_im2col_kernel);
    size += tensor3D_get_sizeof_mem_allocated(l->cache.conv.dL_dW_im2col_output);
    size += matrix_get_sizeof_mem_allocated(l->cache.conv.dL_dW_im2col_output_sum);
    size += tensor3D_get_sizeof_mem_allocated(l->cache.conv.delta_im2col_input);
    size += matrix_get_sizeof_mem_allocated(l->cache.conv.delta_im2col_kernel);
    size += tensor3D_get_sizeof_mem_allocated(l->cache.conv.delta_im2col_output);

    return size;
}

unsigned long layer_flatten_get_sizeof_mem_allocated(Layer* l) {
    unsigned long size = 0;
    size += sizeof(l);
    size += sizeof(l->params);
    size += sizeof(l->params.flat);
    size += sizeof(l->cache.flat);
    size += matrix_get_sizeof_mem_allocated(l->cache.flat.output);
    size += matrix_get_sizeof_mem_allocated(l->cache.flat.delta);

    return size;
}

unsigned long layer_max_pool_get_sizeof_mem_allocated(Layer* l) {
    unsigned long size = 0;
    size += sizeof(l);
    size += sizeof(l->params);
    size += sizeof(l->params.max_pool);
    size += sizeof(l->cache.max_pool);  
    size += tensor4D_get_sizeof_mem_allocated(l->cache.max_pool.output);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.max_pool.delta);
    size += tensor4D_uint16_get_sizeof_mem_allocated(l->cache.max_pool.argmax);

    return size;
}

unsigned long layer_batch_norm_conv2D_get_sizeof_mem_allocated(Layer* l) {
    unsigned long size = 0;
    size += sizeof(l);
    size += sizeof(l->params);
    size += sizeof(l->params.bn_conv);
    size += sizeof(l->cache.bn_conv);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.bn_conv.output);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.bn_conv.z);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.bn_conv.delta);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.bn_conv.dL_dA);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.bn_conv.dA_dZ);
    size += tensor4D_get_sizeof_mem_allocated(l->cache.bn_conv.x_normalized);
    size += matrix_get_sizeof_mem_allocated(l->cache.bn_conv.mean);
    size += matrix_get_sizeof_mem_allocated(l->cache.bn_conv.variance);
    size += matrix_get_sizeof_mem_allocated(l->cache.bn_conv.running_mean);
    size += matrix_get_sizeof_mem_allocated(l->cache.bn_conv.running_variance);
    size += matrix_get_sizeof_mem_allocated(l->cache.bn_conv.gamma);
    size += matrix_get_sizeof_mem_allocated(l->cache.bn_conv.beta);
    size += matrix_get_sizeof_mem_allocated(l->cache.bn_conv.gamma_grad);
    size += matrix_get_sizeof_mem_allocated(l->cache.bn_conv.beta_grad);

    return size;
}