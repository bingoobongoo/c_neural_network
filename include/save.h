#pragma once

#include <stdio.h>

#include "config.h"

void save_epoch_to_csv(
    int epoch, 
    nn_float loss, 
    nn_float train_acc, 
    nn_float val_acc, 
    nn_float time,
    char* csv_filename
);