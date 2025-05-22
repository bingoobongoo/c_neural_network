#pragma once

#include "config.h"
#include "matrix.h"

typedef struct {
    nn_float accuracy;
    int batch_size;
    Matrix* y_pred_argmax;
    Matrix* y_true_argmax;
    
} Score;

Score* score_new(int batch_size);
void score_free(Score* s);
void score_batch(Score* self, Matrix* y_pred, Matrix* y_true);
void update_confusion_matrix(Score* self, Matrix* y_pred, Matrix* y_true, Matrix* conf_m);