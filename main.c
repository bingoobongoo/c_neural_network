#include "network.h"
#include <time.h>

int main() {
    srand(time(NULL));

    NeuralNet* net = neural_net_new(SIGMOID, 0.0, 32);
    add_input_layer(5, net);
    add_deep_layer(3, net);
    add_deep_layer(3, net);
    add_output_layer(1, net);
    neural_net_compile(net);

    neural_net_info(net);
    int n_layers = net->n_in_layers + net->n_ou_layers + net->n_de_layers;
    for (int i=1; i<n_layers; i++) {
        matrix_print(net->layers[i]->weights);
        printf("\n");
    }
    neural_net_free(net);
}
