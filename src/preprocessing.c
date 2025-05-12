#include "preprocessing.h"

Matrix* one_hot_encode(Matrix* column, int n_classes) {
    Matrix* one_hot = matrix_new(column->n_rows, n_classes);
    matrix_fill(one_hot, 0.0);
    for (int i=0; i<column->n_rows; i++) {
        int class = (int)matrix_get(column, i, 0);
        matrix_assign(one_hot, i, class, 1.0);
    }

    return one_hot;
}

void normalize(Matrix* m) {
    float min = matrix_get(m, 0, 0);
    float max = matrix_get(m, 0, 0);

    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            if (matrix_get(m, i, j) < min)
                min = matrix_get(m, i, j);
            if (matrix_get(m, i, j) > max)
                max = matrix_get(m, i, j);
        }
    }

    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            float x = matrix_get(m, i, j);
            matrix_assign(m, i, j, (x - min)/(max - min));
        }
    }
}

void renormalize(Matrix* m, int original_min, int original_max) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            float x = matrix_get(m, i, j);
            matrix_assign(m, i, j, x*(original_max - original_min) + original_min);
        }
    }
}

void shuffle_matrix_inplace(Matrix* feature_m, Matrix* label_m) {
    if (feature_m->n_rows != label_m->n_rows) {
        fprintf(stderr, "shuffle_data_inplace: Mismatched row counts\n");
        exit(1);
    }

    int f_cols = feature_m->n_cols;
    int l_cols = label_m->n_cols;

    for (int i = feature_m->n_rows - 1; i > 0; i--) {
        int j = rand() % (i+1);

        for (int k=0; k<f_cols; k++) {
            float tmp = matrix_get(feature_m, i, k);
            matrix_assign(feature_m, i, k, matrix_get(feature_m, j, k));
            matrix_assign(feature_m, j, k, tmp);
        }

        for (int k=0; k<l_cols; k++) {
            float tmp = matrix_get(label_m, i, k);
            matrix_assign(label_m, i, k, matrix_get(label_m, j, k));
            matrix_assign(label_m, j, k, tmp);
        }
    }
}

void shuffle_tensor4D_inplace(Tensor4D* feature_t, Matrix* label_m) {
    if (feature_t->n_filters != label_m->n_rows) {
        fprintf(stderr, "shuffle_data_inplace: Mismatched row counts\n");
        exit(1);
    }

    int l_cols = label_m->n_cols;

    for (int i=feature_t->n_filters-1; i>0; i--) {
        int j= rand() % (i+1);

        Tensor3D* tmp_f = feature_t->filters[i];
        feature_t->filters[i] = feature_t->filters[j];
        feature_t->filters[j] = tmp_f;

        for (int k=0; k<l_cols; k++) {
            float tmp = matrix_get(label_m, i, k);
            matrix_assign(label_m, i, k, matrix_get(label_m, j, k));
            matrix_assign(label_m, j, k, tmp);
        }
    }
}