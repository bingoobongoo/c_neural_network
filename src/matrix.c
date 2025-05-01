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

void matrix_free_view(Matrix* view) {
    if (view == NULL) return;

    free(view->entries);
    view->entries = NULL;

    free(view);
    view = NULL;
}

void matrix_save(Matrix* m, char* file_path) {
    FILE* file = fopen(file_path, "w");

    fprintf(file, "%d\n", m->n_rows);
    fprintf(file, "%d\n", m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            fprintf(file, "%.6lf ", m->entries[i][j]);
        }
        fprintf(file, "\n");
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

void matrix_copy_into(Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            into->entries[i][j] = m->entries[i][j];
        }
    }
}

void matrix_assign(Matrix** to, Matrix* from) {
    Matrix* old = *to;
    *to = from;
    matrix_free(old);
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

Matrix* matrix_slice_rows(Matrix* m, int start_idx, int slice_size) {
    if (start_idx >= m->n_rows) {
        printf("Index out of range");
        exit(1);
    }

    Matrix* slice = matrix_new(slice_size, m->n_cols);
    if (start_idx + slice_size > m->n_rows) {
        slice_size = m->n_rows - start_idx;
    }

    for (int i=0; i<slice_size; i++) {
        for (int j=0; j<m->n_cols; j++) {
            slice->entries[i][j] = m->entries[i + start_idx][j];
        }
    }

    return slice;
}

void matrix_slice_rows_into(Matrix* m, int start_idx, int slice_size, Matrix* into) {
    if (start_idx >= m->n_rows) {
        printf("Index out of range");
        exit(1);
    }
    if (start_idx + slice_size > m->n_rows) {
        slice_size = m->n_rows - start_idx;
    }

    for (int i=0; i<slice_size; i++) {
        for (int j=0; j<m->n_cols; j++) {
            into->entries[i][j] = m->entries[i + start_idx][j];
        }
    }
}

bool matrix_check_dimensions(Matrix* m1, Matrix* m2) {
    if (m1->n_rows == m2->n_rows && m1->n_cols == m2->n_cols)
        return true;

    return false;
}

Matrix* matrix_add(Matrix* m1, Matrix* m2) {
    if (!matrix_check_dimensions(m1, m2)) {
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

void matrix_add_into(Matrix* m1, Matrix* m2, Matrix* into) {
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            double sum = m1->entries[i][j] + m2->entries[i][j];
            into->entries[i][j] = sum;
        }
    }
}

Matrix* matrix_subtract(Matrix* m1, Matrix* m2) {
    if (!matrix_check_dimensions(m1, m2)) {
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

void matrix_subtract_into(Matrix* m1, Matrix* m2, Matrix* into) {
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            double diff = m1->entries[i][j] - m2->entries[i][j];
            into->entries[i][j] = diff;
        }
    }
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

void matrix_dot_into(Matrix* m1, Matrix* m2, Matrix* into) {
    if (m1->n_cols != m2->n_rows) {
        printf("Matrices have wrong dimensions: ");
        matrix_print_dimensions(m1); printf(" and "); matrix_print_dimensions(m2);
        exit(1);
    }

    int m = m1->n_rows;
    int k = m1->n_cols;
    int n = m2->n_cols;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            into->entries[i][j] = 0.0;

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < m; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < k; kk += BLOCK_SIZE) {

                for (int i = ii; i < ii + BLOCK_SIZE && i < m; i++) {
                    for (int j = jj; j < jj + BLOCK_SIZE && j < n; j++) {
                        double sum = into->entries[i][j];

                        for (int l = kk; l < kk + BLOCK_SIZE && l < k; l++) {
                            sum += m1->entries[i][l] * m2->entries[l][j];
                        }

                        into->entries[i][j] = sum;
                    }
                }

            }
        }
    }
}

Matrix* matrix_multiply(Matrix* m1, Matrix* m2) {
    if (!matrix_check_dimensions(m1, m2)) {
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

void matrix_multiply_into(Matrix* m1, Matrix* m2, Matrix* into) {
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            double product = m1->entries[i][j] * m2->entries[i][j];
            into->entries[i][j] = product;
        }
    }
}

Matrix* matrix_divide(Matrix* m1, Matrix* m2) {
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    Matrix* quotient_mat = matrix_new(m1->n_rows, m1->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            double quotient = m1->entries[i][j] / m2->entries[i][j];
            quotient_mat->entries[i][j] = quotient;
        }
    }

    return quotient_mat;
}

void matrix_divide_into(Matrix* m1, Matrix* m2, Matrix* into) {
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            double quotient = m1->entries[i][j] / m2->entries[i][j];
            into->entries[i][j] = quotient;
        }
    }
}

Matrix* matrix_sum_axis(Matrix* m, int axis) {
    switch (axis)
    {
    case 0: {
        Matrix* sum_m = matrix_new(1, m->n_rows);
        for (int i=0; i<m->n_rows; i++) {
            double sum = 0.0;
            for (int j=0; j<m->n_cols; j++) {
                sum += m->entries[i][j];
            }
            sum_m->entries[0][i] = sum;
        }

        return sum_m;
        break;
    }
    
    case 1: {
        Matrix* sum_m = matrix_new(1, m->n_cols);
        for (int i=0; i<m->n_cols; i++) {
            double sum = 0.0;
            for (int j=0; j<m->n_rows; j++) {
                sum += m->entries[j][i];
            }
            sum_m->entries[0][i] = sum;
        }

        return sum_m;
        break;
    }
    
    default:
        printf("Invalid axis argument");
        exit(1);
        break;
    }
}

void matrix_sum_axis_into(Matrix* m, int axis, Matrix* into) {
    switch (axis)
    {
    case 0: {
        for (int i=0; i<m->n_rows; i++) {
            double sum = 0.0;
            for (int j=0; j<m->n_cols; j++) {
                sum += m->entries[i][j];
            }
            into->entries[0][i] = sum;
        }

        break;
    }
    
    case 1: {
        for (int i=0; i<m->n_cols; i++) {
            double sum = 0.0;
            for (int j=0; j<m->n_rows; j++) {
                sum += m->entries[j][i];
            }
            into->entries[0][i] = sum;
        }

        break;
    }
    
    default:
        printf("Invalid axis argument");
        exit(1);
        break;
    }
}

double matrix_sum(Matrix* m) {
    double sum = 0;
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            sum += m->entries[i][j];
        }
    }

    return sum;
}

double matrix_average(Matrix* m) {
    return matrix_sum(m) / (double)(m->n_rows * m->n_cols);
}

Matrix* matrix_multiplicate(Matrix* m, int axis, int n_size) {
    switch (axis)
    {
    case 0: {
        Matrix* new_m = matrix_new(m->n_rows, n_size * m->n_cols);
        for (int i=0; i<m->n_rows; i++) {
            for (int n=0; n<n_size; n++) {
                for (int j=0; j<m->n_cols; j++) {
                    new_m->entries[i][n*m->n_cols + j] = m->entries[i][j];
                }
            }
        }
        
        return new_m;
        break;
    }
    
    case 1: {
        Matrix* new_m = matrix_new(n_size * m->n_rows, m->n_cols);
        for (int i=0; i<m->n_cols; i++) {
            for (int n=0; n<n_size; n++) {
                for (int j=0; j<m->n_rows; j++) {
                    new_m->entries[n*m->n_rows + j][i] = m->entries[j][i];
                }
            }
        }

        return new_m;
        break;
    }

    default:
        printf("Invalid axis argument");
        exit(1);
        break;
    }
}

void matrix_argmax_into(Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        int argmax = 0;
        for (int j=0; j<m->n_cols; j++) {
            if (m->entries[i][j] > m->entries[i][argmax])
                argmax = j; 
        }
        into->entries[0][i] = (double)argmax;
    }
}

void matrix_multiplicate_into(Matrix* m, int axis, int n_size, Matrix* into) {
    switch (axis)
    {
    case 0: {
        for (int i=0; i<m->n_rows; i++) {
            for (int n=0; n<n_size; n++) {
                for (int j=0; j<m->n_cols; j++) {
                    into->entries[i][n*m->n_cols + j] = m->entries[i][j];
                }
            }
        }
        
        break;
    }
    
    case 1: {
        for (int i=0; i<m->n_cols; i++) {
            for (int n=0; n<n_size; n++) {
                for (int j=0; j<m->n_rows; j++) {
                    into->entries[n*m->n_rows + j][i] = m->entries[j][i];
                }
            }
        }

        break;
    }

    default:
        printf("Invalid axis argument");
        exit(1);
        break;
    }
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

void matrix_apply_into(double (*func)(double), Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            into->entries[i][j] = func(m->entries[i][j]);
        }
    }
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

void matrix_scale_into(double scalar, Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            into->entries[i][j] = m->entries[i][j] * scalar;
        }
    }
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

void matrix_add_scalar_into(double scalar, Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            into->entries[i][j] = m->entries[i][j] + scalar;
        }
    }
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

void matrix_transpose_into(Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            into->entries[j][i] = m->entries[i][j];
        }
    }
}

void matrix_correlate_into(Matrix* input, Matrix* kernel, Matrix* into, CorrelationType type) {
    switch (type)
    {
    case VALID: {
        int out_h = input->n_rows - kernel->n_rows + 1;
        int out_w = input->n_cols - kernel->n_cols + 1;

        for (int i=0; i<out_h; i++) {
            for (int j=0; j<out_w; j++) {
                double sum = 0.0;
                for (int k=0; k<kernel->n_rows; k++) {
                    for (int l=0; l<kernel->n_cols; l++) {
                        sum += input->entries[i+k][j+l] *
                               kernel->entries[k][l];
                    }
                }
                into->entries[i][j] = sum;
            }
        }
        break;
    }
    
    case FULL: {
        int out_h = input->n_rows + kernel->n_rows - 1;
        int out_w = input->n_cols + kernel->n_cols - 1;
        int input_h_idx, input_w_idx;
        int k, k_stick_out, l, l_stick_out;
        double sum;

        for (int i=0; i<out_h; i++) {
            for (int j=0; j<out_w; j++) {
                sum = 0.0;
                k = kernel->n_rows - i - 1;
                if (k<0) k=0;
                k_stick_out = i - kernel->n_cols;
                if (k_stick_out<0) k_stick_out=0;
                for (k; k<kernel->n_rows - k_stick_out; k++) {
                    l = kernel->n_cols - j - 1;
                    if (l<0) l=0;
                    l_stick_out = j - kernel->n_cols;
                    if (l_stick_out<0) l_stick_out=0;
                    for (l; l<kernel->n_cols - l_stick_out; l++) {
                        input_h_idx = i + k - kernel->n_rows + 1;
                        input_w_idx = j + l - kernel->n_cols + 1;
                        sum += input->entries[input_h_idx][input_w_idx] *
                               kernel->entries[k][l]; 
                    }
                }
                into->entries[i][j] = sum;
            }
        }
        break;
    }
    
    default:
        printf("Correlation type doesn't exist.");
        exit(1);
        break;
    }
}

void matrix_convolve_into(Matrix* input, Matrix* kernel, Matrix* into, CorrelationType type) {
    switch (type)
    {
    case VALID: {
        int out_h = input->n_rows - kernel->n_rows + 1;
        int out_w = input->n_cols - kernel->n_cols + 1;

        for (int i=0; i<out_h; i++) {
            for (int j=0; j<out_w; j++) {
                double sum = 0.0;
                for (int k=0; k<kernel->n_rows; k++) {
                    for (int l=0; l<kernel->n_cols; l++) {
                        sum += input->entries[i+k][j+l] *
                               kernel->entries[kernel->n_rows-k-1][kernel->n_cols-l-1];
                    }
                }
                into->entries[i][j] = sum;
            }
        }
        break;
    }
    
    case FULL: {
        int out_h = input->n_rows + kernel->n_rows - 1;
        int out_w = input->n_cols + kernel->n_cols - 1;
        int input_h_idx, input_w_idx;
        int k, k_stick_out, l, l_stick_out;
        double sum;

        for (int i=0; i<out_h; i++) {
            for (int j=0; j<out_w; j++) {
                sum = 0.0;
                k = kernel->n_rows - i - 1;
                if (k<0) k=0;
                k_stick_out = i - kernel->n_cols;
                if (k_stick_out<0) k_stick_out=0;
                for (k; k<kernel->n_rows - k_stick_out; k++) {
                    l = kernel->n_cols - j - 1;
                    if (l<0) l=0;
                    l_stick_out = j - kernel->n_cols;
                    if (l_stick_out<0) l_stick_out=0;
                    for (l; l<kernel->n_cols - l_stick_out; l++) {
                        input_h_idx = i + k - kernel->n_rows + 1;
                        input_w_idx = j + l - kernel->n_cols + 1;
                        sum += input->entries[input_h_idx][input_w_idx] *
                               kernel->entries
                               [kernel->n_rows-k-1]
                               [kernel->n_cols-l-1]; 
                    }
                }
                into->entries[i][j] = sum;
            }
        }
        break;
    }
    
    default:
        printf("Correlation type doesn't exist.");
        exit(1);
        break;
    }
}