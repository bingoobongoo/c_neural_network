#include "vectorization.h"

inline void simd_add(
    const nn_float* a,
    const nn_float* b,
    nn_float* c,
    int n
) {
    int i=0;
    for (; i+NN_SIMD_WIDTH<=n; i+=NN_SIMD_WIDTH) {
        simd_vec va = SIMD_LOAD(a + i);
        simd_vec vb = SIMD_LOAD(b + i);
        simd_vec vc = SIMD_ADD(va, vb);
        SIMD_STORE(c + i, vc);
    }

    for (; i<n; i++) {
        c[i] = a[i] + b[i];
    }
}

inline void simd_sub(
    const nn_float* a,
    const nn_float* b,
    nn_float* c,
    int n
) {
    int i=0;
    for (; i+NN_SIMD_WIDTH<=n; i+=NN_SIMD_WIDTH) {
        simd_vec va = SIMD_LOAD(a + i);
        simd_vec vb = SIMD_LOAD(b + i);
        simd_vec vc = SIMD_SUB(va, vb);
        SIMD_STORE(c + i, vc);
    }

    for (; i<n; i++) {
        c[i] = a[i] - b[i];
    } 
}

inline void simd_mul(
    const nn_float* a,
    const nn_float* b,
    nn_float* c,
    int n
) {
    int i=0;
    for (; i+NN_SIMD_WIDTH<=n; i+=NN_SIMD_WIDTH) {
        simd_vec va = SIMD_LOAD(a + i);
        simd_vec vb = SIMD_LOAD(b + i);
        simd_vec vc = SIMD_MUL(va, vb);
        SIMD_STORE(c + i, vc);
    }

    for (; i<n; i++) {
        c[i] = a[i] * b[i];
    } 
}

inline void simd_div(
    const nn_float* a,
    const nn_float* b,
    nn_float* c,
    int n
) {
    int i=0;
    for (; i+NN_SIMD_WIDTH<=n; i+=NN_SIMD_WIDTH) {
        simd_vec va = SIMD_LOAD(a + i);
        simd_vec vb = SIMD_LOAD(b + i);
        simd_vec vc = SIMD_DIV(va, vb);
        SIMD_STORE(c + i, vc);
    }

    for (; i<n; i++) {
        c[i] = a[i] / b[i];
    } 
}

inline void simd_scale(
    const nn_float* a,
    nn_float s,
    nn_float* c,
    int n
) {
    simd_vec vs = SIMD_SET1(s);
    int i=0;
    for (; i+NN_SIMD_WIDTH<=n; i+=NN_SIMD_WIDTH) {
        simd_vec va = SIMD_LOAD(a + i);
        simd_vec vc = SIMD_MUL(va, vs);
        SIMD_STORE(c + i, vc);
    }

    for (; i<n; i++) {
        c[i] = a[i] * s;
    } 
}

inline void simd_add_scalar(
    const nn_float* a,
    nn_float s,
    nn_float* c,
    int n
) {
    simd_vec vs = SIMD_SET1(s);
    int i=0;
    for (; i+NN_SIMD_WIDTH<=n; i+=NN_SIMD_WIDTH) {
        simd_vec va = SIMD_LOAD(a + i);
        simd_vec vc = SIMD_ADD(va, vs);
        SIMD_STORE(c + i, vc);
    }

    for (; i<n; i++) {
        c[i] = a[i] + s;
    }
}

inline nn_float simd_sum(
    const nn_float* a,
    int n
) {
    simd_vec vacc = SIMD_SET1((nn_float)0.0);
    nn_float sum = (nn_float)0.0;

    int i=0;
    for (; i+NN_SIMD_WIDTH<=n; i+=NN_SIMD_WIDTH) {
        simd_vec va = SIMD_LOAD(a + i);
        vacc = SIMD_ADD(vacc, va);
    }

    nn_float tmp[NN_SIMD_WIDTH];
    SIMD_STORE(tmp, vacc);
    for (int k=0; k<NN_SIMD_WIDTH; k++) {
        sum += tmp[k];
    }

    for (; i<n; i++) {
        sum += a[i];
    }

    return sum;
}

inline nn_float simd_dot(
    const nn_float* a,
    const nn_float* b,
    int n
) {
    simd_vec vacc = SIMD_SET1((nn_float)0.0);
    nn_float sum = (nn_float)0.0;

    int i=0;
    for (; i+NN_SIMD_WIDTH<=n; i+=NN_SIMD_WIDTH) {
        simd_vec va = SIMD_LOAD(a + i);
        simd_vec vb = SIMD_LOAD(b + i);
        simd_vec vm = SIMD_DIV(va, vb);
        SIMD_ADD(vacc, vm); 
    }

    nn_float tmp[NN_SIMD_WIDTH];
    SIMD_STORE(tmp, vacc);
    for (int j=0; j<NN_SIMD_WIDTH; j++) {
        sum += tmp[j];
    }

    for (; i<n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}