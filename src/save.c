#include "save.h"

void save_epoch_to_csv(
    int epoch, 
    nn_float loss, 
    nn_float train_acc, 
    nn_float val_acc, 
    nn_float time,
    char* csv_filename
) {
    FILE* csv_ptr;

    if (epoch == 1) {
        csv_ptr = fopen(csv_filename, "w");
        fprintf(csv_ptr, "epoch, loss, train_acc, val_acc, time\n");
    }
    else {
        csv_ptr = fopen(csv_filename, "a");
    }

    fprintf(csv_ptr, "%d, %f, %f, %f, %f\n", epoch, loss, train_acc, val_acc, time);
    fclose(csv_ptr);
}