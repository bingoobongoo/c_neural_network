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
        if (layer->cache.conv.filter_flip != NULL) {
            tensor4D_free(layer->cache.conv.filter_flip);
            layer->cache.conv.filter_flip = NULL;
        }
        if (layer->cache.conv.bias != NULL) {
            matrix_free(layer->cache.conv.bias);
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
            matrix_free(layer->cache.conv.bias_gradient);
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
        if (layer->cache.conv.padding != NULL) {
            tensor4D_free(layer->cache.conv.padding);
            layer->cache.conv.padding = NULL;
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
        break;

    case MAX_POOL: 
        if (layer->cache.max_pool.output != NULL) {
            tensor4D_free(layer->cache.conv.output);
            layer->cache.conv.output = NULL;
        }
        if (layer->cache.max_pool.delta != NULL) {
            tensor4D_free(layer->cache.conv.delta);
            layer->cache.conv.delta = NULL;
        }
        if (layer->cache.max_pool.argmax != NULL) {
            tensor4D_uint16_free(layer->cache.max_pool.argmax);
            layer->cache.max_pool.argmax = NULL;
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
        return layer->params.conv.n_units;
        break;

    case MAX_POOL:
        return layer->params.max_pool.n_units;
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
        return layer->cache.conv.output;
        break;

    case MAX_POOL:
        return layer->cache.max_pool.output;
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
        return layer->cache.conv.delta;
        break;

    case MAX_POOL:
        return layer->cache.max_pool.delta;
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
    l->cache.dense.weight_gradient = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias_gradient = matrix_new(
        1,
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

void layer_output_compile(Layer* l, Cost* cost, int batch_size) {
    if (layer_get_n_units(l) == 1)
        l->activation = activation_new(SIGMOID, (nn_float)0.0);
    if (layer_get_n_units(l) > 1)
        l->activation = activation_new(SOFTMAX, (nn_float)0.0);
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
    l->cache.dense.weight_gradient = matrix_new(
        layer_get_n_units(l->prev_layer),
        layer_get_n_units(l)
    );
    l->cache.dense.bias_gradient = matrix_new(
        1,
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
    l->params.conv.output_height = output_height;
    l->params.conv.output_width = output_width;

    l->activation = activation_new(
        act_type,
        act_param
    );
    l->cache.conv.filter = tensor4D_new(
        l->params.conv.filter_size,
        l->params.conv.filter_size,
        l->params.conv.n_filter_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.filter_flip = tensor4D_new(
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
    l->cache.conv.filter_gradient = tensor4D_new(
        l->params.conv.filter_size,
        l->params.conv.filter_size,
        l->params.conv.n_filter_channels,
        l->params.conv.n_filters
    );
    l->cache.conv.bias_gradient = matrix_new(
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

    int pad_h = output_height + 2*(filter_height-1);
    int pad_w = output_width + 2*(filter_width-1);
    l->cache.conv.padding = tensor4D_new(
        pad_h,
        pad_w,
        l->params.conv.n_filters,
        batch_size
    );


    l->cache.conv.fp_im2col_input = matrix_new(
        output_height * output_width,
        filter_height * filter_width * l->params.conv.n_filter_channels
    );
    l->cache.conv.fp_im2col_kernel = matrix_new(
        filter_height * filter_width * l->params.conv.n_filter_channels,
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
        l->params.conv.n_filter_channels
    );
    l->cache.conv.delta_im2col_output = matrix_new(
        input_height * input_width,
        l->params.conv.n_filter_channels
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
            l->cache.conv.filter,
            (nn_float)0.0,
            sqrtf((nn_float)2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer)))
        );

        #endif
        #ifdef DOUBLE_PRECISION

        tensor4D_fill_normal_distribution(
            l->cache.conv.filter,
            (nn_float)0.0,
            sqrt((nn_float)2.0/(layer_get_n_units(l) + layer_get_n_units(l->prev_layer)))
        );
        
        #endif
        break;
    
    case RELU:
    case LRELU:
    case ELU:
        #ifdef SINGLE_PRECISION

        tensor4D_fill_normal_distribution(
            l->cache.conv.filter,
            (nn_float)0.0,
            sqrtf((nn_float)2.0/layer_get_n_units(l->prev_layer))
        );

        #endif
        #ifdef DOUBLE_PRECISION

        tensor4D_fill_normal_distribution(
            l->cache.conv.filter,
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
    
    #ifdef IM2COL
    kernel_into_im2col_fwise(
        l->cache.conv.filter,
        false,
        l->cache.conv.fp_im2col_kernel
    );
    #endif
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

    l->params.max_pool.output_height = output_height;
    l->params.max_pool.output_width = output_width;

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

    l->cache.max_pool.dCost_dA = l->cache.max_pool.delta;

    l->params.max_pool.n_units = 
        l->cache.max_pool.output->n_rows *
        l->cache.max_pool.output->n_cols *
        l->cache.max_pool.output->n_channels;
}

void layer_input_fp(Layer* l, Batch* train_batch, int batch_size) {
    l->cache.dense.output = train_batch->data.matrix;
}

void layer_conv2D_input_fp(Layer* l, Batch* train_batch, int batch_size) {
    l->cache.conv.output = train_batch->data.tensor;
}

void layer_deep_fp(Layer* l, int batch_size) {
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

void layer_output_fp(Layer* l, Batch* label_batch, int batch_size) {
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

void layer_conv2D_fp(Layer* l, int batch_size) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* filter = l->cache.conv.filter;
    Tensor4D* z = l->cache.conv.z;
    Matrix* bias = l->cache.conv.bias;
    Tensor4D* output = l->cache.conv.output;

    #ifdef IM2COL_CONV
    
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
            l->cache.conv.fp_im2col_output,
            false,
            false
        );
        matrix_into_tensor3D(
            l->cache.conv.fp_im2col_output,
            z->filters[n],
            true
        );
        for (int i=0; i<filter->n_filters; i++) {
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
    for (int n=0; n<batch_size; n++) {
        for (int i=0; i<filter->n_filters; i++) {
            matrix_zero(z->filters[n]->channels[i]);
            tensor3D_acc_correlate_into(
                input->filters[n],
                filter->filters[i],
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

void layer_flatten_fp(Layer* l, int batch_size) {
    Tensor4D* t = layer_get_output_tensor4D(l->prev_layer);
    Matrix* m = l->cache.flat.output;
    tensor4D_into_matrix_fwise(
        t,
        m,
        false,
        false
    );
}

void layer_max_pool_fp(Layer* l, int batch_size) {
    Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
    Tensor4D* output = layer_get_output_tensor4D(l);
    Tensor4D_uint16* argmax = l->cache.max_pool.argmax;

    const int filter_size = l->params.max_pool.filter_size;
    const int stride = l->params.max_pool.stride;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n=0; n<batch_size; n++) {
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
    matrix_dot_into(
        layer_get_output_matrix(l->prev_layer), 
        l->cache.dense.delta, 
        l->cache.dense.weight_gradient,
        true,
        false
    );

    // bias gradient (dCost_dB) calculations:
    matrix_sum_axis_into(
        l->cache.dense.delta, 
        1, 
        l->cache.dense.bias_gradient
    );
}

void layer_deep_bp(Layer* l, int batch_size) {
    // delta gradient (dCost_dZ) calculations:
    matrix_dot_into(
        l->next_layer->cache.dense.delta, 
        l->next_layer->cache.dense.weight, 
        l->cache.dense.dCost_dA,
        false,
        true
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
    matrix_dot_into(
        layer_get_output_matrix(l->prev_layer), 
        l->cache.dense.delta, 
        l->cache.dense.weight_gradient,
        true,
        false
    );

    // bias gradient (dCost_dB) calculations:
    matrix_sum_axis_into(
        l->cache.dense.delta, 
        1, 
        l->cache.dense.bias_gradient
    );
}

void layer_conv2D_bp(Layer* l, int batch_size) {
    Tensor4D* filter = l->cache.conv.filter;
    Tensor4D* filter_grad = l->cache.conv.filter_gradient;
    Matrix* bias_grad = l->cache.conv.bias_gradient;
    Tensor4D* delta = l->cache.conv.delta;

    // delta gradient (dCost_dZ) calculation
    if (l->next_layer->l_type == FLATTEN) {
        layer_conv2D_bp_delta_from_flatten(l, batch_size);
    }
    else if (l->next_layer->l_type == MAX_POOL) {
        layer_conv2d_bp_delta_from_max_pool(l, batch_size);
    }
    else if (l->next_layer->l_type == CONV_2D) {
        layer_conv2d_bp_delta_from_conv2d(l, batch_size);
    }


    // filter gradient (dCost_dW) calculation
    {
        Tensor4D* input = layer_get_output_tensor4D(l->prev_layer);
        
        #ifdef IM2COL_CONV

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
                l->cache.conv.dCost_dW_im2col_output,
                false,
                false
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

        #else

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i=0; i<filter->n_filters; i++) {
            for (int j=0; j<filter->n_channels; j++) {
                matrix_zero(filter_grad->filters[i]->channels[j]);
                for (int n=0; n<batch_size; n++) {
                    matrix_acc_correlate_into(
                        input->filters[n]->channels[j],
                        delta->filters[n]->channels[i],
                        filter_grad->filters[i]->channels[j],
                        l->params.conv.stride,
                        VALID
                    );
                }
            }
        }

        #endif
    }

    // bias gradient (dCost_dB) calculation
    for (int i=0; i<filter->n_filters; i++) {
        nn_float sum = (nn_float)0.0;
        for (int n=0; n<batch_size; n++) {
            sum += matrix_sum(delta->filters[n]->channels[i]);
        }
        matrix_assign(bias_grad, 0, i, sum);
    }
}

void layer_conv2D_bp_delta_from_flatten(Layer* l, int batch_size) {
    Tensor4D* delta = l->cache.conv.delta;
    Tensor4D* dA_dZ = l->cache.conv.dActivation_dZ;
    Tensor4D* dCost_dA = l->cache.conv.dCost_dA;
    Tensor4D* z = l->cache.conv.z;

    // #pragma omp parallel for collapse(2) schedule(static)
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

void layer_conv2d_bp_delta_from_max_pool(Layer* l, int batch_size) {
    Tensor4D* delta = l->cache.conv.delta;
    Tensor4D* dA_dZ = l->cache.conv.dActivation_dZ;
    Tensor4D* dCost_dA = l->cache.conv.dCost_dA;
    Tensor4D* z = l->cache.conv.z;
    Tensor4D* output = layer_get_output_tensor4D(l);    
    Tensor4D* delta_next = layer_get_delta_tensor4D(l->next_layer);
    Tensor4D* output_next = layer_get_output_tensor4D(l->next_layer);
    Tensor4D_uint16* argmax = l->next_layer->cache.max_pool.argmax;
    int filter_size = l->next_layer->params.max_pool.filter_size;
    int stride = l->next_layer->params.max_pool.stride;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n=0; n<batch_size; n++) {
        for (int c=0; c<output_next->n_channels; c++) {
            Matrix* dCost_dA_mat = dCost_dA->filters[n]->channels[c];
            matrix_zero(dCost_dA_mat);
            Matrix* delta_next_mat = delta_next->filters[n]->channels[c];
            Matrix_uint16* argmax_mat = argmax->filters[n]->channels[c];
            int out_h = output->n_rows;
            int out_w = output->n_cols;
            int out_h_next = output_next->n_rows;
            int out_w_next = output_next->n_cols;
            for (int i=0; i<out_h_next; i++) {
                for (int j=0; j<out_w_next; j++) {
                    int grad_idx = argmax_mat->entries[i*out_w_next + j];
                    int ker_h_offset = grad_idx / filter_size;
                    int ker_w_offset = grad_idx % filter_size;
                    int in_row = i*stride + ker_h_offset;
                    int in_col = j*stride + ker_w_offset;
                    dCost_dA_mat->entries[in_row*out_w + in_col] +=
                        delta_next_mat->entries[i*out_w_next + j];
                }
            }

        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
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

void layer_conv2d_bp_delta_from_conv2d(Layer* l, int batch_size) {
    Tensor4D* delta = l->cache.conv.delta;
    Tensor4D* dA_dZ = l->cache.conv.dActivation_dZ;
    Tensor4D* dCost_dA = l->cache.conv.dCost_dA;
    Tensor4D* z = l->cache.conv.z;
    Tensor4D* delta_next = l->next_layer->cache.conv.delta;
    Tensor4D* filter_next = l->next_layer->cache.conv.filter;
    Tensor4D* filter_flip_next = l->next_layer->cache.conv.filter_flip;
    Tensor4D* padding = l->next_layer->cache.conv.padding;

    #ifdef IM2COL_CONV

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
            l->next_layer->cache.conv.delta_im2col_output,
            false,
            false
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

    #else

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n=0; n<batch_size; n++) {
        for (int i=0; i<delta->n_channels; i++) {
            matrix_zero(dCost_dA->filters[n]->channels[i]);
            for (int f=0; f<filter_next->n_filters; f++) {
                matrix_acc_convolve_full_into(
                    delta_next->filters[n]->channels[f],
                    filter_flip_next->filters[f]->channels[i],
                    dCost_dA->filters[n]->channels[i],
                    padding->filters[n]->channels[f]
                );
            }
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

    #endif
}

void layer_flatten_bp(Layer* l, int batch_size) {
    matrix_dot_into(
        l->next_layer->cache.dense.delta,
        l->next_layer->cache.dense.weight,
        l->cache.flat.dCost_dA_matrix,
        false,
        true
    );
    if (l->prev_layer->l_type == CONV_2D) {
        matrix_into_tensor4D(
            l->cache.flat.dCost_dA_matrix,
            l->prev_layer->cache.conv.dCost_dA
        );
    }
    else if (l->prev_layer->l_type == MAX_POOL) {
        matrix_into_tensor4D(
            l->cache.flat.dCost_dA_matrix,
            l->prev_layer->cache.max_pool.dCost_dA
        );        
    }
}

void layer_max_pool_bp(Layer* l, int batch_size) {
    // dCost_dZ calculation
    if (l->next_layer->l_type != FLATTEN) {
        Tensor4D* delta = layer_get_delta_tensor4D(l);
        Tensor4D* delta_next = layer_get_delta_tensor4D(l->next_layer);
        Tensor4D* filter_next = l->next_layer->cache.conv.filter;
        Tensor4D* filter_flip_next = l->next_layer->cache.conv.filter_flip;
        Tensor4D* padding = l->next_layer->cache.conv.padding;

        #ifdef IM2COL_CONV

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
                l->next_layer->cache.conv.delta_im2col_output,
                false,
                false
            );
            matrix_into_tensor3D(
                l->next_layer->cache.conv.delta_im2col_output,
                delta->filters[n],
                true
            );
        }

        #else
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (int n=0; n<batch_size; n++) {
            for (int i=0; i<delta->n_channels; i++) {
                matrix_zero(delta->filters[n]->channels[i]);
                for (int f=0; f<filter_next->n_filters; f++) {
                    matrix_acc_convolve_full_into(
                        delta_next->filters[n]->channels[f],
                        filter_flip_next->filters[f]->channels[i],
                        delta->filters[n]->channels[i],
                        padding->filters[n]->channels[f]
                    );
                }
            }
        }

        #endif
    }
}

void layer_deep_update_weights(Layer* l, Optimizer* opt) {
    opt->update_dense_weights(
        l->cache.dense.weight, 
        l->cache.dense.weight_gradient, 
        opt, 
        l->layer_idx
    );
    opt->update_dense_bias(
        l->cache.dense.bias, 
        l->cache.dense.bias_gradient, 
        opt, 
        l->layer_idx
    );
}

void layer_conv2D_update_weights(Layer* l, Optimizer* opt) {
    Tensor4D* filter = l->cache.conv.filter;
    Tensor4D* filter_flip = l->cache.conv.filter_flip;
    Matrix* bias = l->cache.conv.bias;
    Tensor4D* filter_grad = l->cache.conv.filter_gradient;
    Matrix* bias_grad = l->cache.conv.bias_gradient;
    opt->update_conv_weights(
        filter,
        filter_grad,
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
        filter,
        filter_flip
    );

    #ifdef IM2COL
    kernel_into_im2col_fwise(filter, false, l->cache.conv.fp_im2col_kernel);
    #endif
}