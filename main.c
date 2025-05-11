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
    matrix_assign_ptr(&y_train, one_hot_encode(y_train, 10));
    matrix_assign_ptr(&y_test, one_hot_encode(y_test, 10));

    shuffle_matrix_inplace(x_train, y_train);
    shuffle_matrix_inplace(x_test, y_test);
    
    NeuralNet* net = neural_net_new(
        optimizer_sgd_new(0.01),
        RELU, 1.0,
        CAT_CROSS_ENTROPY, 
        32
    );

    add_input_layer(x_train->n_cols, net);
    add_deep_layer(300, net);
    add_deep_layer(100, net);
    add_output_layer(y_train->n_cols, net);

    neural_net_compile(net);
    neural_net_info(net);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    fit(x_train, y_train, 5, 0.1, net);
    gettimeofday(&end, NULL);
    double fit_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    printf("Fit time taken: %.3f seconds\n", fit_time);

    score(x_test, y_test, net);
    confusion_matrix(x_test, y_test, net);

    matrix_free(x_train); matrix_free(y_train);
    matrix_free(x_test); matrix_free(y_test);

    neural_net_free(net);
}
