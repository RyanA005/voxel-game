#ifndef NEURAL_SIMD_H
#define NEURAL_SIMD_H

#include <stdint.h>

void neural_simd_init(void);
const char *nn_simd_kernel_name(void);

float nn_dot_fp32(int n, const float *a, const float *b);
float nn_dot_int8_row(int n, const int8_t *w, float scale, const float *x);

/* Post-ReLU hidden layers: quantize activations once, VNNI int8 dot per row. */
void nn_dense_int8_row_vnni(const float *in, float *out, int in_n, int out_n,
                            const int8_t *w, const float *scales, const float *b,
                            int use_relu);

void nn_dense_int8_row_simd(const float *in, float *out, int in_n, int out_n,
                            const int8_t *w, const float *scales, const float *b,
                            int use_relu);

void nn_dense_fp32_simd(const float *in, float *out, int in_n, int out_n,
                        const float *w, const float *b, int use_relu);

#endif
