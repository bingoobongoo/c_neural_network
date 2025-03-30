#include "network.h"
#include "load_data.h"
#include "preprocessing.h"
#include <time.h>

int main() {
    Matrix* feature_m = load_ubyte_images("data/train-images-idx3-ubyte");
    normalize(feature_m);

    Matrix* label_m = load_ubyte_labels("data/train-labels-idx1-ubyte");
    Matrix* label_one_hot = one_hot_encode(label_m, 10);
    matrix_free(label_m);

    matrix_save(feature_m, "features.mat");
    matrix_save(label_one_hot, "label.mat");

    matrix_free(feature_m);
    matrix_free(label_one_hot);
}
