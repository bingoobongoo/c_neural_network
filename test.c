#include "network.h"
#include "load_data.h"
#include "preprocessing.h"

int main() {
    srand(time(NULL));
    Matrix* x_train = load_ubyte_images("data/mnist/train-images-idx3-ubyte");
    Matrix* x_test = load_ubyte_images("data/mnist/test-images-idx3-ubyte");
    normalize(x_train); 
    normalize(x_test);

    Matrix* y_train = load_ubyte_labels("data/mnist/train-labels-idx1-ubyte");
    Matrix* y_test = load_ubyte_labels("data/mnist/test-labels-idx1-ubyte");
    matrix_assign(&y_train, one_hot_encode(y_train, 10));
    matrix_assign(&y_test, one_hot_encode(y_test, 10));

    shuffle_matrix_inplace(x_train, y_train);
    shuffle_matrix_inplace(x_test, y_test);
    
    NeuralNet* net = neural_net_new(
        optimizer_sgd_new(0.001),
        RELU, 1.0,
        CAT_CROSS_ENTROPY,
        32
    );

    add_conv_input_layer(28, 28, 1, net);
    add_conv_layer(32, 8, 1, net);
    add_conv_layer(64, 10, 1, net);

    neural_net_compile(net);
    neural_net_info(net);
    
    Tensor4D* x_train_tensor = matrix_to_tensor4D(x_train, 28, 28, 1);
    batchify_tensor_into(x_train_tensor, 0, net->train_batch);
    batchify_matrix_into(y_train, 0, net->label_batch);
    forward_prop(net, true);

    matrix_free(x_train); matrix_free(y_train);
    matrix_free(x_test); matrix_free(y_test);
    tensor4D_free(x_train_tensor);

    neural_net_free(net);
}