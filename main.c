#include "network.h"
#include "load_data.h"
#include "preprocessing.h"
#include <time.h>

int main() {
    Matrix* feature_m = load_ubyte_images("data/test-images-idx3-ubyte");
    Matrix* label_m = load_ubyte_labels("data/test-labels-idx1-ubyte");

    shuffle_data_inplace(feature_m, label_m);
    normalize(feature_m);
    Matrix* label_one_hot = one_hot_encode(label_m, 10);
    matrix_free(label_m);

    NeuralNet* net = neural_net_new(RELU, MSE, 0.0, 10);
    add_input_layer(feature_m->n_cols, net);
    add_deep_layer(100, net);
    add_deep_layer(50, net);
    add_output_layer(label_one_hot->n_cols, net);

    neural_net_compile(net);
    neural_net_info(net);

    int start_idx = 0;
    for (int i=0; i<feature_m->n_rows; i+=net->batch_size, start_idx += net->batch_size) {
        Batch test_batch = batchify(feature_m, start_idx, net->batch_size, true);
        forward_prop(&test_batch, net);
        batch_free(&test_batch);
    }

    matrix_free(feature_m);
    matrix_free(label_one_hot);

    neural_net_free(net);
}
