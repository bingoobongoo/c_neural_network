#include "matrix.h"

Matrix* matrix_new(int n_rows, int n_cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->n_rows = n_rows;
    m->n_cols = n_cols;
    m->entries = (float*)malloc(n_rows * n_cols * sizeof(float));

    return m;
}

void matrix_free(Matrix* m) {
    if (m == NULL) return;

    free(m->entries);
    m->entries = NULL;

    free(m);
    m = NULL;
}

float matrix_get(Matrix* m, int row, int col) {
    #ifdef DEBUG

    if (col >= m->n_cols || row >= m->n_rows) {
        printf("Out of bounds error while accessing matrix.");
        exit(1);
    }

    #endif

    return m->entries[row*m->n_cols + col];
}

void matrix_assign(Matrix* m, int row, int col, float num) {
    #ifdef DEBUG

    if (col >= m->n_cols || row >= m->n_rows) {
        printf("Out of bounds error while accessing matrix.");
        exit(1);
    }

    #endif

    m->entries[row*m->n_cols + col] = num;
}

void matrix_save(Matrix* m, char* file_path) {
    FILE* file = fopen(file_path, "w");

    fprintf(file, "%d\n", m->n_rows);
    fprintf(file, "%d\n", m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            fprintf(file, "%.6f ", matrix_get(m, i, j));
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
            float num;
            fscanf(file, "%f", &num);
            matrix_assign(m, i, j, num);
        }
    }

    fclose(file);
    return m;
}

Matrix* matrix_copy(Matrix* m) {
    Matrix* copy = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(copy, i, j, matrix_get(m, i, j));
        }
    }

    return copy;
}

void matrix_copy_into(Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, i, j, matrix_get(m, i, j));
        }
    }
}

void matrix_assign_ptr(Matrix** to, Matrix* from) {
    Matrix* old = *to;
    *to = from;
    matrix_free(old);
}

void matrix_print(Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        printf("%c", '|');
        for (int j=0; j<m->n_cols; j++) {
            if (matrix_get(m, i, j) < 0.0){
                printf("%.3f  ", matrix_get(m, i, j));
            }
            else {
                printf(" %.3f  ", matrix_get(m, i, j));
            }
        }
        printf("%c\n", '|');
    }
}

void matrix_print_dimensions(Matrix* m) {
    printf("[%d x %d]", m->n_rows, m->n_cols);
}

void matrix_zero(Matrix* m) {
    memset(m->entries, 0, m->n_rows * m->n_cols * sizeof(float));
}

void matrix_fill(Matrix* m, float num) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(m, i, j, num);
        }
    }
}

void matrix_fill_normal_distribution(Matrix* m, float mean, float std_deviation) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            float u1, u2;
            do {
                u1 = rand() / (RAND_MAX + 1.0);
            } while (u1 < 1e-10);
            u2 = rand() / (RAND_MAX + 1.0);
        
            float z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
            float x = mean + std_deviation * z;
            matrix_assign(m, i, j, x);
        }
    }
}

Matrix* matrix_slice_rows(Matrix* m, int start_idx, int slice_size) {
    #ifdef DEBUG

    if (start_idx >= m->n_rows) {
        printf("Index out of range");
        exit(1);
    }

    #endif

    Matrix* slice = matrix_new(slice_size, m->n_cols);
    if (start_idx + slice_size > m->n_rows) {
        slice_size = m->n_rows - start_idx;
    }

    for (int i=0; i<slice_size; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(slice, i, j, matrix_get(m, i+start_idx, j));
        }
    }

    return slice;
}

void matrix_slice_rows_into(Matrix* m, int start_idx, int slice_size, Matrix* into) {
    #ifdef DEBUG
    
    if (start_idx >= m->n_rows) {
        printf("Index out of range");
        exit(1);
    }

    #endif

    if (start_idx + slice_size > m->n_rows) {
        slice_size = m->n_rows - start_idx;
    }

    for (int i=0; i<slice_size; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, i, j, matrix_get(m, i+start_idx, j));
        }
    }
}

bool matrix_check_dimensions(Matrix* m1, Matrix* m2) {
    if (m1->n_rows == m2->n_rows && m1->n_cols == m2->n_cols)
        return true;

    return false;
}

Matrix* matrix_add(Matrix* m1, Matrix* m2) {
    #ifdef DEBUG

    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif
    
    Matrix* sum_matrix = matrix_new(m1->n_rows, m1->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            float sum = matrix_get(m1, i, j) + matrix_get(m2, i, j);
            matrix_assign(sum_matrix, i, j, sum);
        }
    }

    return sum_matrix;
}

void matrix_add_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEBUG

    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            float sum = matrix_get(m1, i, j) + matrix_get(m2, i, j);
            matrix_assign(into, i, j, sum);;
        }
    }
}

Matrix* matrix_subtract(Matrix* m1, Matrix* m2) {
    #ifdef DEBUG
    
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    Matrix* diff_matrix = matrix_new(m1->n_rows, m1->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            float diff = matrix_get(m1, i, j) - matrix_get(m2, i, j);
            matrix_assign(diff_matrix, i, j, diff);
        }
    }

    return diff_matrix;
}

void matrix_subtract_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEBUG

    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            float diff = matrix_get(m1, i, j) - matrix_get(m2, i, j);
            matrix_assign(into, i, j, diff);
        }
    }
}

Matrix* matrix_dot(Matrix* m1, Matrix* m2) {
    #ifdef DEBUG

    if (m1->n_cols != m2->n_rows) {
        printf("Matrices have wrong dimensions: ");
        matrix_print_dimensions(m1); printf(" and "); matrix_print_dimensions(m2);
        exit(1);
    }

    #endif

    Matrix* dot_matrix = matrix_new(m1->n_rows, m2->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m2->n_cols; j++) {
            float dot = 0;
            for (int k=0; k<m1->n_cols; k++) {
                dot += matrix_get(m1, i, k) * matrix_get(m2, k, j);
            }
            matrix_assign(dot_matrix, i, j, dot);
        }
    }

    return dot_matrix;
}

void matrix_dot_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEBUG
    
    if (m1->n_cols != m2->n_rows) {
        printf("Matrices have wrong dimensions: ");
        matrix_print_dimensions(m1); printf(" and "); matrix_print_dimensions(m2);
        exit(1);
    }

    #endif
    #ifdef BLAS

    float alpha = 1.0f;
    float beta = 0.0f;

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m1->n_rows,
        m2->n_cols,
        m1->n_cols,
        alpha,
        m1->entries,
        m1->n_cols,
        m2->entries,
        m2->n_cols,
        beta,
        into->entries,
        m2->n_cols
    );

    #endif
    #ifndef BLAS

    matrix_zero(into);
    #pragma omp parallel for collapse(1)
    for (int i=0; i<m1->n_rows; i++) {
        for (int k=0; k<m1->n_cols; k++) {
            for (int j=0; j<m2->n_cols; j++) {
                matrix_assign(
                    into, 
                    i, 
                    j,
                    matrix_get(into, i, j) + matrix_get(m1, i, k) * matrix_get(m2, k, j)
                );
            }
        }
    }

    #endif
}

Matrix* matrix_multiply(Matrix* m1, Matrix* m2) {
    #ifdef DEBUG
    
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif
    
    Matrix* product_matrix = matrix_new(m1->n_rows, m1->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            float product = matrix_get(m1, i, j) * matrix_get(m2, i, j);
            matrix_assign(product_matrix, i, j, product);
        }
    }

    return product_matrix;
}

void matrix_multiply_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEFINE
    
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            float product = matrix_get(m1, i, j) * matrix_get(m2, i, j);
            matrix_assign(into, i, j, product);
        }
    }
}

Matrix* matrix_divide(Matrix* m1, Matrix* m2) {
    #ifdef DEBUG
    
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    Matrix* quotient_mat = matrix_new(m1->n_rows, m1->n_cols);
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            float quotient = matrix_get(m1, i, j) / matrix_get(m2, i, j);
            matrix_assign(quotient_mat, i, j, quotient);
        }
    }

    return quotient_mat;
}

void matrix_divide_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEBUG
    
    if (!matrix_check_dimensions(m1, m2)) {
        printf("Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            float quotient = matrix_get(m1, i, j) / matrix_get(m2, i, j);
            matrix_assign(into, i, j, quotient);
        }
    }
}

Matrix* matrix_sum_axis(Matrix* m, int axis) {
    switch (axis)
    {
    case 0: {
        Matrix* sum_m = matrix_new(1, m->n_rows);
        for (int i=0; i<m->n_rows; i++) {
            float sum = 0.0;
            for (int j=0; j<m->n_cols; j++) {
                sum += matrix_get(m, i, j);
            }
            matrix_assign(sum_m, 0, i, sum);
        }

        return sum_m;
        break;
    }
    
    case 1: {
        Matrix* sum_m = matrix_new(1, m->n_cols);
        for (int i=0; i<m->n_cols; i++) {
            float sum = 0.0;
            for (int j=0; j<m->n_rows; j++) {
                sum += matrix_get(m, j, i);
            }
            matrix_assign(sum_m, 0, i, sum);
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
            float sum = 0.0;
            for (int j=0; j<m->n_cols; j++) {
                sum += matrix_get(m, i, j);
            }
            matrix_assign(into, 0, i, sum);
        }

        break;
    }
    
    case 1: {
        for (int i=0; i<m->n_cols; i++) {
            float sum = 0.0;
            for (int j=0; j<m->n_rows; j++) {
                sum += matrix_get(m, j, i);
            }
            matrix_assign(into, 0, i, sum);
        }

        break;
    }
    
    default:
        printf("Invalid axis argument");
        exit(1);
        break;
    }
}

float matrix_sum(Matrix* m) {
    float sum = 0;
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            sum += matrix_get(m, i, j);
        }
    }

    return sum;
}

float matrix_average(Matrix* m) {
    return matrix_sum(m) / (float)(m->n_rows * m->n_cols);
}

Matrix* matrix_multiplicate(Matrix* m, int axis, int n_size) {
    switch (axis)
    {
    case 0: {
        Matrix* new_m = matrix_new(m->n_rows, n_size * m->n_cols);
        for (int i=0; i<m->n_rows; i++) {
            for (int n=0; n<n_size; n++) {
                for (int j=0; j<m->n_cols; j++) {
                    matrix_assign(new_m, i, n*m->n_cols+j, matrix_get(m, i, j));
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
                    matrix_assign(new_m, n*m->n_rows+j, i, matrix_get(m, j, i));
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
            if (matrix_get(m, i, j) > matrix_get(m, i, argmax))
                argmax = j; 
        }
        matrix_assign(into, 0, i, (float)argmax);
    }
}

void matrix_multiplicate_into(Matrix* m, int axis, int n_size, Matrix* into) {
    switch (axis)
    {
    case 0: {
        for (int i=0; i<m->n_rows; i++) {
            for (int n=0; n<n_size; n++) {
                for (int j=0; j<m->n_cols; j++) {
                    matrix_assign(into, i, n*m->n_cols+j, matrix_get(m, i, j));
                }
            }
        }
        
        break;
    }
    
    case 1: {
        for (int i=0; i<m->n_cols; i++) {
            for (int n=0; n<n_size; n++) {
                for (int j=0; j<m->n_rows; j++) {
                    matrix_assign(into, n*m->n_rows+j, i, matrix_get(m, j, i));
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

Matrix* matrix_apply(float (*func)(float), Matrix* m) {
    Matrix* transformed_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(transformed_matrix, i, j, func(matrix_get(m, i, j)));
        }
    }

    return transformed_matrix;
}

void matrix_apply_into(float (*func)(float), Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, i, j, func(matrix_get(m, i, j)));
        }
    }
}

void matrix_apply_inplace(float (*func)(float), Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(m, i, j, func(matrix_get(m, i, j)));
        }
    }
}

Matrix* matrix_scale(float scalar, Matrix* m) {
    Matrix* scaled_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(scaled_matrix, i, j, matrix_get(m, i, j) * scalar);
        }
    }

    return scaled_matrix;
}

void matrix_scale_into(float scalar, Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, i, j, matrix_get(m, i, j) * scalar);
        }
    }
}

void matrix_scale_inplace(float scalar, Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(m, i, j, matrix_get(m, i, j) * scalar);
        }
    }
}

Matrix* matrix_add_scalar(float scalar, Matrix* m) {
    Matrix* scaled_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(scaled_matrix, i, j, matrix_get(m, i, j) + scalar);
        }
    }

    return scaled_matrix;
}

void matrix_add_scalar_into(float scalar, Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, i, j, matrix_get(m, i, j) + scalar);
            
        }
    }
}

void matrix_add_scalar_inplace(float scalar, Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(m, i, j, matrix_get(m, i, j) + scalar);
        }
    }
}

Matrix* matrix_transpose(Matrix* m) {
    Matrix* transposed_matrix = matrix_new(m->n_cols, m->n_rows);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(transposed_matrix, j, i, matrix_get(m, i, j));
        }
    }

    return transposed_matrix;
}

void matrix_transpose_into(Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, j, i, matrix_get(m, i, j));

        }
    }
}

void matrix_correlate_into(Matrix* input, Matrix* kernel, Matrix* into, int stride, CorrelationType type) {
    switch (type)
    {
    case VALID: {
        int out_h = (input->n_rows - kernel->n_rows)/stride + 1;
        int out_w = (input->n_cols - kernel->n_cols)/stride + 1;

        for (int i=0; i<out_h; i++) {
            for (int j=0; j<out_w; j++) {
                float sum = 0.0;
                for (int k=0; k<kernel->n_rows; k++) {
                    for (int l=0; l<kernel->n_cols; l++) {
                        sum += matrix_get(input, i*stride+k, j*stride+l) *
                               matrix_get(kernel, k, l);
                    }
                }
                matrix_assign(into, i, j, sum);
            }
        }
        break;
    }
    
    case FULL: {
        int out_h = (input->n_rows + kernel->n_rows - 2 + stride)/stride;
        int out_w = (input->n_cols + kernel->n_cols - 2 + stride)/stride;
        int input_h_idx, input_w_idx;
        int k, k_stick_out, l, l_stick_out;
        float sum;

        for (int i=0; i<out_h; i++) {
            for (int j=0; j<out_w; j++) {
                sum = 0.0;
                k = kernel->n_rows - i*stride - 1;
                if (k<0) k=0;
                k_stick_out = i*stride - input->n_rows + 1;
                if (k_stick_out<0) k_stick_out=0;
                for (k; k<kernel->n_rows - k_stick_out; k++) {
                    l = kernel->n_cols - j*stride - 1;
                    if (l<0) l=0;
                    l_stick_out = j*stride - input->n_cols + 1;
                    if (l_stick_out<0) l_stick_out=0;
                    for (l; l<kernel->n_cols - l_stick_out; l++) {
                        input_h_idx = i*stride + k - kernel->n_rows + 1;
                        input_w_idx = j*stride + l - kernel->n_cols + 1;
                        sum += matrix_get(input, input_h_idx, input_w_idx) *
                               matrix_get(kernel, k, l);
                    }
                }
                matrix_assign(into, i, j, sum);
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

void matrix_convolve_into(Matrix* input, Matrix* kernel, Matrix* into, int stride, CorrelationType type) {
    switch (type)
    {
    case VALID: {
        int out_h = (input->n_rows - kernel->n_rows)/stride + 1;
        int out_w = (input->n_cols - kernel->n_cols)/stride + 1;

        for (int i=0; i<out_h; i++) {
            for (int j=0; j<out_w; j++) {
                float sum = 0.0;
                for (int k=0; k<kernel->n_rows; k++) {
                    for (int l=0; l<kernel->n_cols; l++) {
                        sum += matrix_get(input, i*stride+k, j*stride+l) *
                               matrix_get(kernel, kernel->n_rows-k-1, kernel->n_cols-l-1);
                    }
                }
                matrix_assign(into, i, j, sum);
            }
        }
        break;
    }
    
    case FULL: {
        int out_h = (input->n_rows + kernel->n_rows - 2 + stride)/stride;
        int out_w = (input->n_cols + kernel->n_cols - 2 + stride)/stride;
        int input_h_idx, input_w_idx;
        int k, k_stick_out, l, l_stick_out;
        float sum;

        for (int i=0; i<out_h; i++) {
            for (int j=0; j<out_w; j++) {
                sum = 0.0;
                k = kernel->n_rows - i*stride - 1;
                if (k<0) k=0;
                k_stick_out = i*stride - input->n_rows + 1;
                if (k_stick_out<0) k_stick_out=0;
                for (k; k<kernel->n_rows - k_stick_out; k++) {
                    l = kernel->n_cols - j*stride - 1;
                    if (l<0) l=0;
                    l_stick_out = j*stride - input->n_cols + 1;
                    if (l_stick_out<0) l_stick_out=0;
                    for (l; l<kernel->n_cols - l_stick_out; l++) {
                        input_h_idx = i*stride + k - kernel->n_rows + 1;
                        input_w_idx = j*stride + l - kernel->n_cols + 1;
                        sum += matrix_get(input, input_h_idx, input_w_idx) *
                               matrix_get(kernel, kernel->n_rows-k-1, kernel->n_cols-l-1);
                    }
                }
                matrix_assign(into, i, j, sum);
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

void matrix_max_pool_into(Matrix* input, Matrix* into, int kernel_size, int stride) {
    int out_h = (input->n_rows - kernel_size)/stride + 1;
    int out_w = (input->n_cols - kernel_size)/stride + 1;
    for (int i=0; i<out_h; i++) {
        for (int j=0; j<out_w; j++) {
            float max = matrix_get(input, i*stride, j*stride);
            for (int k=0; k<kernel_size; k++) {
                for (int l=1; l<kernel_size; l++) {
                    float entry = matrix_get(input, i*stride+k, j*stride+l);
                    if (entry > max) max = entry;
                }
            }
            matrix_assign(into, i, j, max);
        }
    }
}