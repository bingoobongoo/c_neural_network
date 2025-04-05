#include "network.h"
#include "load_data.h"
#include "preprocessing.h"
#include <time.h>
#include <sys/time.h>

int main() {
    srand(time(NULL));
    Matrix* x_train = load_ubyte_images("data/train-images-idx3-ubyte");
    Matrix* x_test = load_ubyte_images("data/test-images-idx3-ubyte");
    normalize(x_train); 
    normalize(x_test);

    Matrix* y_train = load_ubyte_labels("data/train-labels-idx1-ubyte");
    Matrix* y_test = load_ubyte_labels("data/test-labels-idx1-ubyte");
    matrix_assign(&y_train, one_hot_encode(y_train, 10));
    matrix_assign(&y_test, one_hot_encode(y_test, 10));

    shuffle_data_inplace(x_train, y_train);
    shuffle_data_inplace(x_test, y_test);
    
    NeuralNet* net = neural_net_new(
        optimizer_sgd_new(0.0012),
        RELU, 0.0,
        CAT_CROSS_ENTROPY, 
        128
    );

    add_input_layer(x_train->n_cols, net);
    add_deep_layer(392, net);
    add_deep_layer(196, net);
    add_deep_layer(98, net);
    add_output_layer(y_train->n_cols, net);

    neural_net_compile(net);
    neural_net_info(net);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    fit(x_train, y_train, 100, 0.1, net);
    gettimeofday(&end, NULL);
    double fit_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    printf("Fit time taken: %.3f seconds\n", fit_time);

    score(x_test, y_test, net);
    confusion_matrix(x_test, y_test, net);

    matrix_free(x_train); matrix_free(y_train);
    matrix_free(x_test); matrix_free(y_test);

    neural_net_free(net);
}
