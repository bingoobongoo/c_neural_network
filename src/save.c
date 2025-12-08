#include "save.h"

void save_epoch_to_csv(
    int epoch, 
    nn_float train_loss, 
    nn_float val_loss, 
    nn_float train_acc, 
    nn_float val_acc, 
    nn_float time,
    nn_float samples_per_sec,
    char* csv_filename
) {
    FILE* csv_ptr;

    if (epoch == 1) {
        csv_ptr = fopen(csv_filename, "w");
        fprintf(csv_ptr, "epoch,train_loss,val_loss,train_acc,val_acc,time,samples/s\n");
    }
    else {
        csv_ptr = fopen(csv_filename, "a");
    }

    fprintf(
        csv_ptr, 
        "%d,%f,%f,%f,%f,%f,%f\n", 
        epoch, train_loss, val_loss, train_acc, val_acc, time, samples_per_sec);
    fclose(csv_ptr);
}