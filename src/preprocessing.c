#include "preprocessing.h"

Matrix* one_hot_encode(Matrix* column, int n_classes) {
    Matrix* one_hot = matrix_new(column->n_rows, n_classes);
    matrix_fill(one_hot, 0.0);
    for (int i=0; i<column->n_rows; i++) {
        int class = (int)column->entries[i][0];
        one_hot->entries[i][class] = 1.0;
    }

    return one_hot;
}

void normalize(Matrix* m) {
    double min = m->entries[0][0];
    double max = m->entries[0][0];

    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            if (m->entries[i][j] < min)
                min = m->entries[i][j];
            if (m->entries[i][j] > max)
                max = m->entries[i][j];
        }
    }

    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            double x = m->entries[i][j];
            m->entries[i][j] = (x - min)/(max - min);
        }
    }
}

void renormalize(Matrix* m, int original_min, int original_max) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            double x = m->entries[i][j];
            m->entries[i][j] = x*(original_max - original_min) + original_min;
        }
    }
}

void shuffle_data_inplace(Matrix* feature_m, Matrix* label_m) {
    if (feature_m->n_rows != label_m->n_rows) {
        fprintf(stderr, "shuffle_data_inplace: Mismatched row counts\n");
        exit(1);
    }

    for (int i = feature_m->n_rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        double* tmp_f = feature_m->entries[i];
        feature_m->entries[i] = feature_m->entries[j];
        feature_m->entries[j] = tmp_f;

        double* tmp_l = label_m->entries[i];
        label_m->entries[i] = label_m->entries[j];
        label_m->entries[j] = tmp_l;
    }
}