#include "score.h"

Score* score_new(int batch_size) {
    Score* s = (Score*)malloc(sizeof(Score));
    s->accuracy = -1.0;
    s->batch_size = batch_size;
    s->y_pred_argmax = matrix_new(1, batch_size);
    s->y_true_argmax = matrix_new(1, batch_size);

    return s;
}

void score_free(Score* s) {
    matrix_free(s->y_pred_argmax);
    matrix_free(s->y_true_argmax);
    free(s);
}

void score_batch(Score* self, Matrix* y_pred, Matrix* y_true) {
    matrix_argmax_into(y_pred, self->y_pred_argmax);
    matrix_argmax_into(y_true, self->y_true_argmax);
    double n_samples = self->y_pred_argmax->n_cols;
    double correct_preds = 0.0;
    for (int i=0; i<n_samples; i++) {
        if (self->y_pred_argmax->entries[0][i] == self->y_true_argmax->entries[0][i]) {
            correct_preds++;
        }
    }

    self->accuracy = correct_preds / n_samples;
}