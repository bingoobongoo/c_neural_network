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
    float n_samples = self->y_pred_argmax->n_cols;
    float correct_preds = 0.0;
    for (int i=0; i<n_samples; i++) {
        if (matrix_get(self->y_pred_argmax,0,i) == matrix_get(self->y_true_argmax,0,i)) {
            correct_preds++;
        }
    }

    self->accuracy = correct_preds / n_samples;
}

void update_confusion_matrix(Score* self, Matrix* y_pred, Matrix* y_true, Matrix* c) {
    matrix_argmax_into(y_pred, self->y_pred_argmax);
    matrix_argmax_into(y_true, self->y_true_argmax);
    float n_samples = self->y_pred_argmax->n_cols;
    for (int i=0; i<n_samples; i++) {
        int true_class = matrix_get(self->y_true_argmax, 0, i);
        int pred_class = matrix_get(self->y_pred_argmax, 0, i);
        float count = matrix_get(c, true_class, pred_class);
        matrix_assign(c, true_class, pred_class, count+1);
    }
}