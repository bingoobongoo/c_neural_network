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

void update_confusion_matrix(Score* self, Matrix* y_pred, Matrix* y_true, Matrix* c) {
    matrix_argmax_into(y_pred, self->y_pred_argmax);
    matrix_argmax_into(y_true, self->y_true_argmax);
    double n_samples = self->y_pred_argmax->n_cols;
    for (int i=0; i<n_samples; i++) {
        int true_class = self->y_true_argmax->entries[0][i];
        int pred_class = self->y_pred_argmax->entries[0][i];
        c->entries[true_class][pred_class]++;
    }
    Matrix* row_sum = matrix_sum_axis(c, 0);
    for (int i=0; i<row_sum->n_cols; i++) {
        for (int j=0; j<row_sum->n_cols; j++) {
            c->entries[i][j] = c->entries[i][j] / row_sum->entries[0][i];
        }
    }

    matrix_free(row_sum);
}