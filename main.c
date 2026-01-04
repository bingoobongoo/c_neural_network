#include "network.h"
#include "load_data.h"

int main() {
    openblas_set_num_threads(4);
    srand(time(NULL));

    Matrix* x_train = load_ubyte_images("data/fashion_mnist/train-images-idx3-ubyte");
    Matrix* x_test = load_ubyte_images("data/fashion_mnist/test-images-idx3-ubyte");
    Matrix* y_train = load_ubyte_labels("data/fashion_mnist/train-labels-idx1-ubyte");
    Matrix* y_test = load_ubyte_labels("data/fashion_mnist/test-labels-idx1-ubyte");
    
    normalize(x_train); 
    normalize(x_test);

    Matrix* y_train_onehot = one_hot_encode(y_train, 10);
    Matrix* y_test_onehot = one_hot_encode(y_test, 10);

    shuffle_matrix_inplace(x_train, y_train_onehot);
    shuffle_matrix_inplace(x_test, y_test_onehot);
    
    NeuralNet* net = neural_net_new(
        optimizer_adam_new(0.001, 0.9, 0.999),
        RELU, 0.01,
        CAT_CROSS_ENTROPY, 
        32
    );

    add_input_layer(x_train->n_cols, net);
    add_dense_layer(1000, net);
    add_dense_layer(500, net);
    add_dense_layer(300, net);
    add_dense_layer(100, net);
    add_output_layer(y_train_onehot->n_cols, net);
    neural_net_compile(net);
    
    neural_net_info(net);
    
    fit(x_train, y_train_onehot, 10, 0.1, net);
    score(x_test, y_test_onehot, net);
    confusion_matrix(x_test, y_test_onehot, net);

    matrix_free(x_train); matrix_free(y_train);
    matrix_free(x_test); matrix_free(y_test);
    matrix_free(y_train_onehot); matrix_free(y_test_onehot);
    neural_net_free(net);
}
