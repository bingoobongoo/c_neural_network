#include "network.h"
#include "load_data.h"
#include "preprocessing.h"
#include <time.h>
#include <sys/time.h>

int main() {
    srand(time(NULL));
    Matrix* feature_m = load_ubyte_images("data/train-images-idx3-ubyte");
    Matrix* label_m = load_ubyte_labels("data/train-labels-idx1-ubyte");

    shuffle_data_inplace(feature_m, label_m);
    normalize(feature_m);
    Matrix* label_one_hot = one_hot_encode(label_m, 10);
    matrix_free(label_m);

    NeuralNet* net = neural_net_new(ELU, MSE, 1.0, 256, 0.001);
    add_input_layer(feature_m->n_cols, net);
    add_deep_layer(300, net);
    add_deep_layer(100, net);
    add_output_layer(label_one_hot->n_cols, net);

    neural_net_compile(net);
    neural_net_info(net);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    fit(feature_m, label_one_hot, 5, net);
    gettimeofday(&end, NULL);
    double fit_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Fit time taken: %.3f seconds\n", fit_time);

    matrix_free(feature_m);
    matrix_free(label_one_hot);

    neural_net_free(net);
}
