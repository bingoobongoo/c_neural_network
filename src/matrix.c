#include "matrix.h"

Matrix* matrix_new(int n_rows, int n_cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->n_rows = n_rows;
    m->n_cols = n_cols;
    m->entries = (double**)malloc(n_rows * sizeof(double*));
    for (int i=0; i<n_rows; i++) {
        m->entries[i] = malloc(n_cols * sizeof(double));
    }

    return m;
}

void matrix_free(Matrix* m) {
    if (m == NULL) return;

    for (int i=0; i<m->n_rows; i++) {
        free(m->entries[i]);
        m->entries[i] = NULL;
    }

    free(m->entries);
    m->entries = NULL;

    free(m);
    m = NULL;
}

void matrix_save(Matrix* m, char* file_path) {
    FILE* file = fopen(file_path, "w");

    fprintf(file, "%d\n", m->n_rows);
    fprintf(file, "%d\n", m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            fprintf(file, "%.6lf\n", m->entries[i][j]);
        }
    }

    fclose(file);
}

Matrix* matrix_load(char* file_path) {
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        printf("File \"%s\" does not exist.", file_path);
        exit(1);
    }

    int n_rows, n_cols;
    fscanf(file, "%d", &n_rows);
    fscanf(file, "%d", &n_cols);

    Matrix* m = matrix_new(n_rows, n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            double num;
            fscanf(file, "%lf", &num);
            m->entries[i][j] = num;
        }
    }

    fclose(file);
    return m;
}

Matrix* matrix_copy(Matrix* m) {
    Matrix* copy = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            copy->entries[i][j] = m->entries[i][j];
        }
    }

    return copy;
}

void matrix_print(Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        printf("%c", '|');
        for (int j=0; j<m->n_cols; j++) {
            if (m->entries[i][j] < 0.0){
                printf("%.3lf  ", m->entries[i][j]);
            }
            else {
                printf(" %.3lf  ", m->entries[i][j]);
            }
        }
        printf("%c\n", '|');
    }
}

void matrix_print_dimensions(Matrix* m) {
    printf("[%d x %d]", m->n_rows, m->n_cols);
}

void matrix_fill(Matrix* m, double num) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            m->entries[i][j] = num;
        }
    }
}

void matrix_fill_normal_distribution(Matrix* m, double mean, double std_deviation) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            double u1, u2;
            do {
                u1 = rand() / (RAND_MAX + 1.0);
            } while (u1 < 1e-10);
            u2 = rand() / (RAND_MAX + 1.0);
        
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
            double x = mean + std_deviation * z;
            m->entries[i][j] = x;
        }
    }
}

Matrix* matrix_flatten(Matrix* m, int axis) {
    if (axis == 0) { // flatten to horizontal vector
        Matrix* flat = matrix_new(1, m->n_rows * m->n_cols);
        for (int i=0; i<m->n_rows; i++) {
            for (int j=0; j<m->n_cols; j++) {
                flat->entries[0][i*m->n_rows + j] = m->entries[i][j];
            }
        }

        return flat;
    }
    if (axis == 1) { // flatten to vertical vector
        Matrix* flat = matrix_new(m->n_rows * m->n_cols, 1);
        for (int i=0; i<m->n_rows; i++) {
            for (int j=0; j<m->n_cols; j++) {
                flat->entries[i*m->n_rows + j][0] = m->entries[i][j];
            }
        }

        return flat;
    }

    printf("No axis %d. 0 for horizontal, 1 for vertical", axis); 
    exit(1);
}

bool check_dimensions(Matrix* m1, Matrix* m2) {
    if (m1->n_rows == m2->n_rows && m1->n_cols == m2->n_cols)
        return true;

    return false;
}

Matrix* matrix_add(Matrix* m1, Matrix* m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }
    
    Matrix* sum_matrix = matrix_new(m1->n_rows, m1->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            double sum = m1->entries[i][j] + m2->entries[i][j];
            sum_matrix->entries[i][j] = sum;
        }
    }

    return sum_matrix;
}

Matrix* matrix_subtract(Matrix* m1, Matrix* m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    Matrix* diff_matrix = matrix_new(m1->n_rows, m1->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            double diff = m1->entries[i][j] - m2->entries[i][j];
            diff_matrix->entries[i][j] = diff;
        }
    }

    return diff_matrix;
}

Matrix* matrix_dot(Matrix* m1, Matrix* m2) {
    if (m1->n_cols != m2->n_rows) {
        printf("Matrices have wrong dimensions: ");
        matrix_print_dimensions(m1); printf(" and "); matrix_print_dimensions(m2);
        exit(1);
    }

    Matrix* dot_matrix = matrix_new(m1->n_rows, m2->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m2->n_cols; j++) {
            double dot = 0;
            for (int k=0; k<m1->n_cols; k++) {
                dot += m1->entries[i][k] * m2->entries[k][j];
            }
            dot_matrix->entries[i][j] = dot;
        }
    }

    return dot_matrix;
}

Matrix* matrix_multiply(Matrix* m1, Matrix* m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    Matrix* product_matrix = matrix_new(m1->n_rows, m1->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            double product = m1->entries[i][j] * m2->entries[i][j];
            product_matrix->entries[i][j] = product;
        }
    }

    return product_matrix;
}

Matrix* matrix_apply(double (*func)(double), Matrix* m) {
    Matrix* transformed_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            transformed_matrix->entries[i][j] = func(m->entries[i][j]);
        }
    }

    return transformed_matrix;
}

void matrix_apply_inplace(double (*func)(double), Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            m->entries[i][j] = func(m->entries[i][j]);
        }
    }
}

Matrix* matrix_scale(double scalar, Matrix* m) {
    Matrix* scaled_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            scaled_matrix->entries[i][j] = m->entries[i][j] * scalar;
        }
    }

    return scaled_matrix;
}

void matrix_scale_inplace(double scalar, Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            m->entries[i][j] *= scalar;
        }
    }
}

Matrix* matrix_add_scalar(double scalar, Matrix* m) {
    Matrix* scaled_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            scaled_matrix->entries[i][j] = m->entries[i][j] + scalar;
        }
    }

    return scaled_matrix;
}

void matrix_add_scalar_inplace(double scalar, Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            m->entries[i][j] += scalar;
        }
    }
}

Matrix* matrix_transpose(Matrix* m) {
    Matrix* transposed_matrix = matrix_new(m->n_cols, m->n_rows);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            transposed_matrix->entries[j][i] = m->entries[i][j];
        }
    }

    return transposed_matrix;
}