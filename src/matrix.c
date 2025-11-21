#include "matrix.h"

Matrix* matrix_new(int n_rows, int n_cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->n_rows = n_rows;
    m->n_cols = n_cols;
    m->entries = (nn_float*)malloc(n_rows * n_cols * sizeof(nn_float));
    m->view = false;

    return m;
}

Matrix* matrix_view_new(int n_rows, int n_cols, nn_float* entries) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->n_rows = n_rows;
    m->n_cols = n_cols;
    m->entries = entries;
    m->view = true;

    return m;
}

void matrix_free(Matrix* m) {
    if (m == NULL) return;

    if (!m->view) free(m->entries);
    m->entries = NULL;

    free(m);
    m = NULL;
}

#ifdef INLINE
inline nn_float matrix_get(Matrix* m, int row, int col) {
    #ifdef DEBUG

    if (col >= m->n_cols || row >= m->n_rows) {
        printf("Out of bounds error while accessing matrix.");
        exit(1);
    }

    #endif

    return m->entries[row*m->n_cols + col];
}
#else
nn_float matrix_get(Matrix* m, int row, int col) {
    #ifdef DEBUG

    if (col >= m->n_cols || row >= m->n_rows) {
        printf("Out of bounds error while accessing matrix.");
        exit(1);
    }

    #endif

    return m->entries[row*m->n_cols + col];
}
#endif

#ifdef INLINE
inline void matrix_assign(Matrix* m, int row, int col, nn_float num) {
    #ifdef DEBUG

    if (col >= m->n_cols || row >= m->n_rows) {
        printf("Out of bounds error while accessing matrix.");
        exit(1);
    }

    #endif

    m->entries[row*m->n_cols + col] = num;
}
#else
void matrix_assign(Matrix* m, int row, int col, nn_float num) {
    #ifdef DEBUG

    if (col >= m->n_cols || row >= m->n_rows) {
        printf("Out of bounds error while accessing matrix.");
        exit(1);
    }

    #endif

    m->entries[row*m->n_cols + col] = num;
}
#endif

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
            nn_float num;
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
            if (matrix_get(m, i, j) < (nn_float)0.0){
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
    memset(m->entries, 0, m->n_rows * m->n_cols * sizeof(nn_float));
}

void matrix_fill(Matrix* m, nn_float num) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(m, i, j, num);
        }
    }
}

void matrix_fill_normal_distribution(Matrix* m, nn_float mean, nn_float std_deviation) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            nn_float u1, u2;
            do {
                u1 = rand() / (RAND_MAX + (nn_float)1.0);
            } while (u1 < (nn_float)1e-10);
            u2 = rand() / (RAND_MAX + (nn_float)1.0);
        
            #ifdef SINGLE_PRECISION
            nn_float z = sqrtf((nn_float)-2.0 * logf(u1)) * cosf((nn_float)2.0 * PI * u2);
            #endif
            #ifdef DOUBLE_PRECISION
            nn_float z = sqrt((nn_float)-2.0 * log(u1)) * cos((nn_float)2.0 * PI * u2);
            #endif

            nn_float x = mean + std_deviation * z;
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
            nn_float sum = matrix_get(m1, i, j) + matrix_get(m2, i, j);
            matrix_assign(sum_matrix, i, j, sum);
        }
    }

    return sum_matrix;
}

void matrix_add_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEBUG

    if (!matrix_check_dimensions(m1, m2)) {
        printf("(matrix_add_into):Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    #ifdef VECTORIZATION
    simd_add(
        m1->entries, 
        m2->entries, 
        into->entries,
        m1->n_rows * m1->n_cols
    );
    #else
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            nn_float sum = matrix_get(m1, i, j) + matrix_get(m2, i, j);
            matrix_assign(into, i, j, sum);
        }
    }
    #endif
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
            nn_float diff = matrix_get(m1, i, j) - matrix_get(m2, i, j);
            matrix_assign(diff_matrix, i, j, diff);
        }
    }

    return diff_matrix;
}

void matrix_subtract_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEBUG

    if (!matrix_check_dimensions(m1, m2)) {
        printf("(matrix_subtract_into):Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    #ifdef VECTORIZATION
    simd_sub(
        m1->entries, 
        m2->entries, 
        into->entries,
        m1->n_rows * m1->n_cols
    );
    #else
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            nn_float diff = matrix_get(m1, i, j) - matrix_get(m2, i, j);
            matrix_assign(into, i, j, diff);
        }
    }
    #endif
}

void matrix_dot_into(Matrix* m1, Matrix* m2, Matrix* into, bool m1_trans, bool m2_trans) {
    #ifdef DEBUG

    bool check_no_trans = (!m1_trans && !m2_trans) && (m1->n_cols != m2->n_rows);
    bool check_m1_trans = m1_trans && !m2_trans && (m1->n_rows != m2->n_rows);
    bool check_m2_trans = m2_trans && !m1_trans && (m1->n_cols != m2->n_cols);
    
    if (check_no_trans || check_m1_trans || check_m2_trans) {
        printf("Matrices have wrong dimensions: ");
        matrix_print_dimensions(m1); printf(" and "); matrix_print_dimensions(m2);
        printf("\nm1_trans=%d   m2_trans=%d\n", m1_trans, m2_trans);
        exit(1);
    }

    #endif
    #ifdef BLAS
    
    nn_float alpha = (nn_float)1.0;
    nn_float beta = (nn_float)0.0;
    CBLAS_TRANSPOSE m1_cblas_trans = (m1_trans) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE m2_cblas_trans = (m2_trans) ? CblasTrans : CblasNoTrans;
    const int M = m1_trans ? m1->n_cols : m1->n_rows;
    const int N = m2_trans ? m2->n_rows : m2->n_cols;
    const int K = m1_trans ? m1->n_rows : m1->n_cols;
    const int lda = (m1_cblas_trans == CblasNoTrans) ? K : M;
    const int ldb = (m2_cblas_trans == CblasNoTrans) ? N : K;
    const int ldc = N;

    #ifdef SINGLE_PRECISION

    cblas_sgemm(
        CblasRowMajor,
        m1_cblas_trans,
        m2_cblas_trans,
        M,
        N,
        K,
        alpha,
        m1->entries,
        lda,
        m2->entries,
        ldb,
        beta,
        into->entries,
        ldc
    );

    #elif defined(DOUBLE_PRECISION)

    cblas_dgemm(
        CblasRowMajor,
        m1_cblas_trans,
        m2_cblas_trans,
        M,
        N,
        K,
        alpha,
        m1->entries,
        lda,
        m2->entries,
        ldb,
        beta,
        into->entries,
        ldc
    );

    #endif
    #endif

    #ifndef BLAS

    matrix_zero(into);

    if (!m1_trans && !m2_trans) {
        #if defined(CACHE_LOCALITY) && !defined(VECTORIZATION)

        #ifdef MULTI_THREADING
        #pragma omp parallel for
        #endif
        for (int i=0; i<m1->n_rows; i++) {
            for (int k=0; k<m1->n_cols; k++) {
                nn_float aik = matrix_get(m1, i, k);
                for (int j=0; j<m2->n_cols; j++) {
                    matrix_assign(
                        into, 
                        i, 
                        j,
                        matrix_get(into, i, j) + aik * matrix_get(m2, k, j)
                    );
                }
            }
        }

        #elif defined(VECTORIZATION)

        #ifdef MULTI_THREADING
        #pragma omp parallel for schedule(static)
        #endif
        for (int i=0; i<m1->n_rows; i++) {
            nn_float* m1_row = m1->entries + i * m1->n_cols;
            nn_float* into_row = into->entries + i * into->n_cols;

            for (int k=0; k<m1->n_cols; k++) {
                nn_float aik = m1_row[k];
                simd_vec vm1 = SIMD_SET1(aik);
                nn_float* m2_row = m2->entries + k * m2->n_cols;
                int j=0;

                for (; j+NN_SIMD_WIDTH<=m2->n_cols; j+=NN_SIMD_WIDTH) {
                    simd_vec vm2 = SIMD_LOAD(m2_row + j);
                    simd_vec vinto = SIMD_LOAD(into_row + j);
                    simd_vec vprod = SIMD_MUL(vm1, vm2);
                    vinto = SIMD_ADD(vinto, vprod);
                    SIMD_STORE(into_row + j, vinto);
                }

                for (; j<m2->n_cols; j++) {
                    into_row[j] += aik * m2_row[j];
                }
            }
        }

        #elif !defined(CACHE_LOCALITY)

        #ifdef MULTI_THREADING
        #pragma omp parallel for
        #endif
        for (int j=0; j<m2->n_cols; j++) {
            for (int k=0; k<m1->n_cols; k++) {
                for (int i=0; i<m1->n_rows; i++) {
                    nn_float aik = matrix_get(m1, i, k);
                    matrix_assign(
                        into,
                        i,
                        j,
                        matrix_get(into, i, j) + aik * matrix_get(m2, k, j)
                    );
                }
            }
        }
        
        #endif
    }
    else {
        printf("Matrix dot only works for m1_trans=false and m2_trans=false without BLAS.");
        exit(1);
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
            nn_float product = matrix_get(m1, i, j) * matrix_get(m2, i, j);
            matrix_assign(product_matrix, i, j, product);
        }
    }

    return product_matrix;
}

void matrix_multiply_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEBUG
    
    if (!matrix_check_dimensions(m1, m2)) {
        printf("(matrix_multiply_into):Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    #ifdef VECTORIZATION
    simd_mul(
        m1->entries, 
        m2->entries, 
        into->entries,
        m1->n_rows * m1->n_cols
    );
    #else
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            nn_float product = matrix_get(m1, i, j) * matrix_get(m2, i, j);
            matrix_assign(into, i, j, product);
        }
    }
    #endif
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
            nn_float quotient = matrix_get(m1, i, j) / matrix_get(m2, i, j);
            matrix_assign(quotient_mat, i, j, quotient);
        }
    }

    return quotient_mat;
}

void matrix_divide_into(Matrix* m1, Matrix* m2, Matrix* into) {
    #ifdef DEBUG
    
    if (!matrix_check_dimensions(m1, m2)) {
        printf("(matrix_divide_into):Matrices have different dimensions: ");
        matrix_print_dimensions(m1); printf(" != "); matrix_print_dimensions(m2);
        exit(1); 
    }

    #endif

    #ifdef VECTORIZATION
    simd_div(
        m1->entries, 
        m2->entries, 
        into->entries,
        m1->n_rows * m1->n_cols
    );
    #else
    for (int i=0; i<m1->n_rows; i++) {
        for (int j=0; j<m1->n_cols; j++) {
            nn_float quotient = matrix_get(m1, i, j) / matrix_get(m2, i, j);
            matrix_assign(into, i, j, quotient);
        }
    }
    #endif
}

Matrix* matrix_sum_axis(Matrix* m, int axis) {
    switch (axis)
    {
    case 0: {
        Matrix* sum_m = matrix_new(1, m->n_rows);
        for (int i=0; i<m->n_rows; i++) {
            nn_float sum = (nn_float)0.0;
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
            nn_float sum = (nn_float)0.0;
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
            nn_float sum = (nn_float)0.0;
            for (int j=0; j<m->n_cols; j++) {
                sum += matrix_get(m, i, j);
            }
            matrix_assign(into, 0, i, sum);
        }

        break;
    }
    
    case 1: {
        for (int i=0; i<m->n_cols; i++) {
            nn_float sum = (nn_float)0.0;
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

nn_float matrix_sum(Matrix* m) {
    nn_float sum = (nn_float)0.0;
    #ifdef VECTORIZATION
    sum = simd_sum(
        m->entries,
        m->n_rows * m->n_cols
    );
    #else
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            sum += matrix_get(m, i, j);
        }
    }
    #endif

    return sum;
}

nn_float matrix_average(Matrix* m) {
    return matrix_sum(m) / (nn_float)(m->n_rows * m->n_cols);
}

nn_float matrix_min(Matrix* m) {
    nn_float min = matrix_get(m, 0, 0);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=1; j<m->n_cols; j++) {
            nn_float x = matrix_get(m, i, j);
            if (x < min) min = x;
        }
    }

    return min;
}

nn_float matrix_max(Matrix* m) {
    nn_float max = matrix_get(m, 0, 0);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=1; j<m->n_cols; j++) {
            nn_float x = matrix_get(m, i, j);
            if (x > max) max = x;
        }
    }

    return max;
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
        matrix_assign(into, 0, i, (nn_float)argmax);
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

Matrix* matrix_apply(nn_float (*func)(nn_float), Matrix* m) {
    Matrix* transformed_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(transformed_matrix, i, j, func(matrix_get(m, i, j)));
        }
    }

    return transformed_matrix;
}

void matrix_apply_into(nn_float (*func)(nn_float), Matrix* m, Matrix* into) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, i, j, func(matrix_get(m, i, j)));
        }
    }
}

void matrix_apply_inplace(nn_float (*func)(nn_float), Matrix* m) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(m, i, j, func(matrix_get(m, i, j)));
        }
    }
}

Matrix* matrix_scale(nn_float scalar, Matrix* m) {
    Matrix* scaled_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(scaled_matrix, i, j, matrix_get(m, i, j) * scalar);
        }
    }

    return scaled_matrix;
}

void matrix_scale_into(nn_float scalar, Matrix* m, Matrix* into) {
    #ifdef VECTORIZATION
    simd_scale(
        m->entries, 
        scalar, 
        into->entries,
        m->n_rows * m->n_cols
    );
    #else
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, i, j, matrix_get(m, i, j) * scalar);
        }
    }
    #endif
}

void matrix_scale_inplace(nn_float scalar, Matrix* m) {
    #ifdef VECTORIZATION
    simd_scale(
        m->entries, 
        scalar, 
        m->entries,
        m->n_rows * m->n_cols
    );
    #else
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(m, i, j, matrix_get(m, i, j) * scalar);
        }
    }
    #endif
}

Matrix* matrix_add_scalar(nn_float scalar, Matrix* m) {
    Matrix* scaled_matrix = matrix_new(m->n_rows, m->n_cols);
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(scaled_matrix, i, j, matrix_get(m, i, j) + scalar);
        }
    }

    return scaled_matrix;
}

void matrix_add_scalar_into(nn_float scalar, Matrix* m, Matrix* into) {
    #ifdef VECTORIZATION
    simd_add_scalar(
        m->entries, 
        scalar, 
        into->entries,
        m->n_rows * m->n_cols
    );
    #else
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(into, i, j, matrix_get(m, i, j) + scalar);
            
        }
    }
    #endif
}

void matrix_add_scalar_inplace(nn_float scalar, Matrix* m) {
    #ifdef VECTORIZATION
    simd_add_scalar(
        m->entries, 
        scalar, 
        m->entries,
        m->n_rows * m->n_cols
    );
    #else
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            matrix_assign(m, i, j, matrix_get(m, i, j) + scalar);
        }
    }
    #endif
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

void matrix_flip_into(Matrix* m, Matrix* into) {
    int ker_h = m->n_rows;
    int ker_w = m->n_cols;
    for (int i=0; i<ker_h; i++) {
        nn_float* src_row = m->entries + i*ker_w;
        nn_float* dst_row = into->entries + (ker_h - 1 - i)*ker_w;
        for (int j=0; j<ker_w; j++) {
            dst_row[ker_w - 1 - j] = src_row[j];
        }
    }
}

void matrix_acc_correlate_into(Matrix* input, Matrix* kernel, Matrix* into, int stride, CorrelationType type) {
    switch (type)
    {
    case VALID: {
        int out_h = (input->n_rows - kernel->n_rows)/stride + 1;
        int out_w = (input->n_cols - kernel->n_cols)/stride + 1;
        int i, j, k, l;
        int is, js;
        nn_float sum, x;

        for (i=0; i<out_h; i++) {
            is = i*stride;
            for (j=0; j<out_w; j++) {
                js = j*stride;
                sum = (nn_float)0.0;
                x = matrix_get(into, i, j);
                for (k=0; k<kernel->n_rows; k++) {
                    for (l=0; l<kernel->n_cols; l++) {
                        sum += matrix_get(input, is+k, js+l) *
                               matrix_get(kernel, k, l);
                    }
                }
                matrix_assign(into, i, j, x+sum);
            }
        }
        break;
    }
    
    case FULL: {
        int out_h = (input->n_rows + kernel->n_rows - 2 + stride)/stride;
        int out_w = (input->n_cols + kernel->n_cols - 2 + stride)/stride;
        int input_h_idx, input_w_idx;
        int i, j, k, k_stick_out, l, l_stick_out;
        int is, js;
        nn_float sum, x;

        for (i=0; i<out_h; i++) {
            is = i*stride;
            for (j=0; j<out_w; j++) {
                js = j*stride;
                sum = (nn_float)0.0;
                x = matrix_get(into, i, j);
                k = kernel->n_rows - is - 1;
                if (k<0) k=0;
                k_stick_out = is - input->n_rows + 1;
                if (k_stick_out<0) k_stick_out=0;
                for (k; k<kernel->n_rows - k_stick_out; k++) {
                    l = kernel->n_cols - js - 1;
                    input_h_idx = is + k - kernel->n_rows + 1;
                    if (l<0) l=0;
                    l_stick_out = js - input->n_cols + 1;
                    if (l_stick_out<0) l_stick_out=0;
                    for (l; l<kernel->n_cols - l_stick_out; l++) {
                        input_w_idx = js + l - kernel->n_cols + 1;
                        sum += matrix_get(input, input_h_idx, input_w_idx) *
                               matrix_get(kernel, k, l);
                    }
                }
                matrix_assign(into, i, j, x+sum);
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

void matrix_acc_convolve_valid_into(Matrix* input, Matrix* kflip, Matrix* into, int stride) {
    int in_h = input->n_rows;
    int in_w = input->n_cols;
    int ker_h = kflip->n_rows;
    int ker_w = kflip->n_cols;
    int out_h = (input->n_rows - ker_h)/stride + 1;
    int out_w = (input->n_cols - ker_w)/stride + 1;
    nn_float sum, x;

    for (int i=0; i<out_h; i++) {
        int is = i*stride;
        nn_float* out_row_ptr = into->entries + i*out_w;

        for (int j=0; j<out_w; j++) {
            int js = j*stride;
            sum = (nn_float)0.0;

            for (int k=0; k<ker_h; k++) {
                nn_float* in_row_ptr = input->entries + (is+k)*in_w + js;
                nn_float* ker_row_ptr = kflip->entries + k*ker_w;

                for (int l=0; l<ker_w; l++) {
                    sum += in_row_ptr[l] * ker_row_ptr[l];
                }
            }
            out_row_ptr[j] += sum;
        }
    }
}

void matrix_acc_convolve_full_into(Matrix* input, Matrix* kflip, Matrix* into, Matrix* padding) {
    int in_h = input->n_rows;
    int in_w = input->n_cols;
    int ker_h = kflip->n_rows;
    int ker_w = kflip->n_cols;
    int pad_h = in_h + 2*(ker_h-1);
    int pad_w = in_w + 2*(ker_w-1);;
    matrix_zero(padding);

    for (int i=0; i<in_h; i++) {
        nn_float* src = input->entries + i*in_w;
        nn_float* dst = padding->entries + (i+ker_h-1)*pad_w + ker_w - 1;
        memcpy(dst, src, in_w*sizeof(nn_float));
    }

    matrix_acc_convolve_valid_into(padding, kflip, into, 1);
}

void matrix_max_pool_into(Matrix* input, Matrix* into, Matrix_uint16* argmax, int kernel_size, int stride) {
    int out_h = (input->n_rows - kernel_size)/stride + 1;
    int out_w = (input->n_cols - kernel_size)/stride + 1;
    int is, js;
    for (int i=0; i<out_h; i++) {
        is = i*stride;
        for (int j=0; j<out_w; j++) {
            js = j*stride;
            nn_float max = -INFINITY;
            uint16_t max_idx = 0;
            for (int k=0; k<kernel_size; k++) {
                for (int l=0; l<kernel_size; l++) {
                    nn_float entry = matrix_get(input, is+k, js+l);
                    if (entry > max) {
                        max = entry;
                        max_idx = k*kernel_size + l;
                    }
                }
            }
            matrix_assign(into, i, j, max);
            argmax->entries[i*argmax->n_cols + j] = max_idx;
        }
    }
}

size_t matrix_get_sizeof_mem_allocated(Matrix* m) {
    size_t size = 0;
    if (m == NULL) return size;

    size += sizeof(*m);

    if (!m->view)
        size += m->n_rows * m->n_cols * sizeof(nn_float);

    return size;
}

Matrix_uint16* matrix_uint16_new(int n_rows, int n_cols) {
    Matrix_uint16* m = (Matrix_uint16*)malloc(sizeof(Matrix_uint16));
    m->n_rows = n_rows;
    m->n_cols = n_cols;
    m->entries = (uint16_t*)malloc(n_rows * n_cols * sizeof(uint16_t));

    return m;
}

void matrix_uint16_free(Matrix_uint16* m) {
    if (m == NULL) return;

    free(m->entries);
    m->entries = NULL;

    free(m);
    m = NULL;
}

void matrix_uint16_fill(Matrix_uint16* m, uint16_t num) {
    for (int i=0; i<m->n_rows; i++) {
        for (int j=0; j<m->n_cols; j++) {
            m->entries[i*m->n_cols+j] = num;
        }
    }   
}