#pragma once

#include <stdio.h>

#include "config.h"

void save_epoch_to_csv(
    int epoch, 
    nn_float train_loss, 
    nn_float val_loss, 
    nn_float train_acc, 
    nn_float val_acc, 
    nn_float time,
    nn_float samples_per_sec,
    char* csv_filename
);