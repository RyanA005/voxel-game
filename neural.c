#include "neural.h"
#include "observation.h"
#include "world.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define USE_AVX2 1
#endif

typedef struct {
    int version, input_dim, h1, h2, output_dim;
    float *w1, *b1, *w2, *b2, *w3, *b3;
    int8_t *q1, *q2, *q3;
    float *s1, *s2, *s3;
    float *buf1, *buf2;
} MLPModel;

static MLPModel model;
static int model_loaded = 0;

#ifdef USE_AVX2
static float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s);
    __m128 sums = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

static void dense_relu_f(const float *restrict in, float *restrict out,
                         int in_n, int out_n,
                         const float *restrict w, const float *restrict b) {
#ifdef USE_AVX2
    for (int o = 0; o < out_n; o++) {
        const float *row = w + (size_t)o * (size_t)in_n;
        __m256 sum = _mm256_set1_ps(b[o]);
        int i = 0;
        for (; i + 7 < in_n; i += 8) {
            __m256 xv = _mm256_loadu_ps(in + i);
            __m256 wv = _mm256_loadu_ps(row + i);
            sum = _mm256_fmadd_ps(wv, xv, sum);
        }
        float total = hsum256(sum);
        for (; i < in_n; i++) total += row[i] * in[i];
        out[o] = total > 0.0f ? total : 0.0f;
    }
#else
    for (int o = 0; o < out_n; o++) {
        const float *row = w + (size_t)o * (size_t)in_n;
        float sum = b[o];
        for (int i = 0; i < in_n; i++) sum += row[i] * in[i];
        out[o] = sum > 0.0f ? sum : 0.0f;
    }
#endif
}

static void dense_linear_f(const float *restrict in, float *restrict out,
                           int in_n, int out_n,
                           const float *restrict w, const float *restrict b) {
#ifdef USE_AVX2
    for (int o = 0; o < out_n; o++) {
        const float *row = w + (size_t)o * (size_t)in_n;
        __m256 sum = _mm256_set1_ps(b[o]);
        int i = 0;
        for (; i + 7 < in_n; i += 8) {
            __m256 xv = _mm256_loadu_ps(in + i);
            __m256 wv = _mm256_loadu_ps(row + i);
            sum = _mm256_fmadd_ps(wv, xv, sum);
        }
        float total = hsum256(sum);
        for (; i < in_n; i++) total += row[i] * in[i];
        out[o] = total;
    }
#else
    for (int o = 0; o < out_n; o++) {
        const float *row = w + (size_t)o * (size_t)in_n;
        float sum = b[o];
        for (int i = 0; i < in_n; i++) sum += row[i] * in[i];
        out[o] = sum;
    }
#endif
}

static void dense_relu_i8(const float *restrict in, float *restrict out,
                          int in_n, int out_n,
                          const int8_t *restrict w, const float *restrict scales,
                          const float *restrict b) {
    for (int o = 0; o < out_n; o++) {
        const int8_t *row = w + (size_t)o * (size_t)in_n;
        float sum = b[o];
        float scale = scales[o];
        int i = 0;
        for (; i + 3 < in_n; i += 4) {
            sum += in[i    ] * (float)row[i    ] * scale;
            sum += in[i + 1] * (float)row[i + 1] * scale;
            sum += in[i + 2] * (float)row[i + 2] * scale;
            sum += in[i + 3] * (float)row[i + 3] * scale;
        }
        for (; i < in_n; i++) sum += in[i] * (float)row[i] * scale;
        out[o] = sum > 0.0f ? sum : 0.0f;
    }
}

static void dense_linear_i8(const float *restrict in, float *restrict out,
                            int in_n, int out_n,
                            const int8_t *restrict w, const float *restrict scales,
                            const float *restrict b) {
    for (int o = 0; o < out_n; o++) {
        const int8_t *row = w + (size_t)o * (size_t)in_n;
        float sum = b[o];
        float scale = scales[o];
        int i = 0;
        for (; i + 3 < in_n; i += 4) {
            sum += in[i    ] * (float)row[i    ] * scale;
            sum += in[i + 1] * (float)row[i + 1] * scale;
            sum += in[i + 2] * (float)row[i + 2] * scale;
            sum += in[i + 3] * (float)row[i + 3] * scale;
        }
        for (; i < in_n; i++) sum += in[i] * (float)row[i] * scale;
        out[o] = sum;
    }
}

static void fuse_normalization(float *input_mean, float *input_std,
                               float *output_mean, float *output_std) {
    int in_n = model.input_dim, h1 = model.h1, h2 = model.h2, out_n = model.output_dim;
    int h_out = h2 > 0 ? h2 : h1;

    for (int o = 0; o < h1; o++) {
        float bias = model.b1[o];
        float *row = model.w1 + (size_t)o * (size_t)in_n;
        for (int i = 0; i < in_n; i++) {
            float inv_std = 1.0f / input_std[i];
            float w = row[i] * inv_std;
            bias -= w * input_mean[i];
            row[i] = w;
        }
        model.b1[o] = bias;
    }

    for (int o = 0; o < out_n; o++) {
        float scale = output_std[o];
        float *row = model.w3 + (size_t)o * (size_t)h_out;
        model.b3[o] = model.b3[o] * scale + output_mean[o];
        for (int j = 0; j < h_out; j++) row[j] *= scale;
    }
    (void)output_mean;
}

static void free_model(void) {
    free(model.w1); free(model.b1); free(model.w2); free(model.b2);
    free(model.w3); free(model.b3);
    free(model.q1); free(model.q2); free(model.q3);
    free(model.s1); free(model.s2); free(model.s3);
    free(model.buf1); free(model.buf2);
    memset(&model, 0, sizeof(model));
    model_loaded = 0;
}

static int read_floats(FILE *f, float *dst, int n) {
    if (n <= 0) return 1;
    return (int)fread(dst, sizeof(float), (size_t)n, f) == (size_t)n;
}

static int read_i8(FILE *f, int8_t *dst, int n) {
    if (n <= 0) return 1;
    return (int)fread(dst, 1, (size_t)n, f) == (size_t)n;
}

static int load_float(FILE *f) {
    int h_out = model.h2 > 0 ? model.h2 : model.h1;
    float *input_mean = (float *)malloc((size_t)model.input_dim * sizeof(float));
    float *input_std = (float *)malloc((size_t)model.input_dim * sizeof(float));
    float *output_mean = (float *)malloc((size_t)model.output_dim * sizeof(float));
    float *output_std = (float *)malloc((size_t)model.output_dim * sizeof(float));

    model.w1 = (float *)malloc((size_t)model.h1 * (size_t)model.input_dim * sizeof(float));
    model.b1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    model.w2 = model.h2 > 0 ? (float *)malloc((size_t)model.h2 * (size_t)model.h1 * sizeof(float)) : NULL;
    model.b2 = model.h2 > 0 ? (float *)malloc((size_t)model.h2 * sizeof(float)) : NULL;
    model.w3 = (float *)malloc((size_t)model.output_dim * (size_t)h_out * sizeof(float));
    model.b3 = (float *)malloc((size_t)model.output_dim * sizeof(float));
    model.buf1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    model.buf2 = model.h2 > 0 ? (float *)malloc((size_t)model.h2 * sizeof(float)) : NULL;

    int ok = input_mean && input_std && output_mean && output_std && model.w1 && model.b1 && model.w3 && model.b3 && model.buf1;
    ok = ok && (model.h2 == 0 || (model.w2 && model.b2 && model.buf2));
    ok = ok && read_floats(f, input_mean, model.input_dim);
    ok = ok && read_floats(f, input_std, model.input_dim);
    ok = ok && read_floats(f, output_mean, model.output_dim);
    ok = ok && read_floats(f, output_std, model.output_dim);
    ok = ok && read_floats(f, model.w1, model.h1 * model.input_dim);
    ok = ok && read_floats(f, model.b1, model.h1);
    ok = ok && read_floats(f, model.w2, model.h2 * model.h1);
    ok = ok && read_floats(f, model.b2, model.h2);
    ok = ok && read_floats(f, model.w3, model.output_dim * h_out);
    ok = ok && read_floats(f, model.b3, model.output_dim);

    if (!ok) {
        free(input_mean); free(input_std); free(output_mean); free(output_std);
        return 0;
    }

    for (int i = 0; i < model.input_dim; i++)
        if (input_std[i] < 1e-6f) input_std[i] = 1.0f;
    for (int i = 0; i < model.output_dim; i++)
        if (output_std[i] < 1e-6f) output_std[i] = 1.0f;

    fuse_normalization(input_mean, input_std, output_mean, output_std);
    free(input_mean); free(input_std); free(output_mean); free(output_std);
    return 1;
}

static int load_int8(FILE *f) {
    int h_out = model.h2 > 0 ? model.h2 : model.h1;

    model.q1 = (int8_t *)malloc((size_t)model.h1 * (size_t)model.input_dim);
    model.s1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    model.b1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    model.q2 = model.h2 > 0 ? (int8_t *)malloc((size_t)model.h2 * (size_t)model.h1) : NULL;
    model.s2 = model.h2 > 0 ? (float *)malloc((size_t)model.h2 * sizeof(float)) : NULL;
    model.b2 = model.h2 > 0 ? (float *)malloc((size_t)model.h2 * sizeof(float)) : NULL;
    model.q3 = (int8_t *)malloc((size_t)model.output_dim * (size_t)h_out);
    model.s3 = (float *)malloc((size_t)model.output_dim * sizeof(float));
    model.b3 = (float *)malloc((size_t)model.output_dim * sizeof(float));
    model.buf1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    model.buf2 = model.h2 > 0 ? (float *)malloc((size_t)model.h2 * sizeof(float)) : NULL;

    int ok = model.q1 && model.s1 && model.b1 && model.q3 && model.s3 && model.b3 && model.buf1;
    ok = ok && (model.h2 == 0 || (model.q2 && model.s2 && model.b2 && model.buf2));
    ok = ok && read_i8(f, model.q1, model.h1 * model.input_dim);
    ok = ok && read_floats(f, model.s1, model.h1);
    ok = ok && read_floats(f, model.b1, model.h1);
    if (model.h2 > 0) {
        ok = ok && read_i8(f, model.q2, model.h2 * model.h1);
        ok = ok && read_floats(f, model.s2, model.h2);
        ok = ok && read_floats(f, model.b2, model.h2);
    }
    ok = ok && read_i8(f, model.q3, model.output_dim * h_out);
    ok = ok && read_floats(f, model.s3, model.output_dim);
    ok = ok && read_floats(f, model.b3, model.output_dim);
    return ok;
}

int neural_load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    free_model();

    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || magic != MODEL_MAGIC) { fclose(f); return 0; }
    if (fread(&version, 4, 1, f) != 1 || (version != MODEL_VER_FLOAT && version != MODEL_VER_INT8)) {
        fclose(f); return 0;
    }

    int dims[4];
    if (fread(dims, sizeof(int), 4, f) != 4) { fclose(f); return 0; }

    model.version = (int)version;
    model.input_dim = dims[0];
    model.h1 = dims[1];
    model.h2 = dims[2];
    model.output_dim = dims[3];

    if (model.input_dim != INPUT_DIM) {
        free_model();
        fclose(f);
        return 0;
    }

    int ok = (version == MODEL_VER_FLOAT) ? load_float(f) : load_int8(f);
    fclose(f);
    if (!ok) { free_model(); return 0; }

    model_loaded = 1;
    return 1;
}

int neural_model_loaded(void) { return model_loaded; }

void neural_forward(const float *input_raw, float *output_raw) {
    if (model.version == MODEL_VER_INT8) {
        dense_relu_i8(input_raw, model.buf1, model.input_dim, model.h1,
                      model.q1, model.s1, model.b1);
        if (model.h2 > 0) {
            dense_relu_i8(model.buf1, model.buf2, model.h1, model.h2,
                          model.q2, model.s2, model.b2);
            dense_linear_i8(model.buf2, output_raw, model.h2, model.output_dim,
                            model.q3, model.s3, model.b3);
        } else {
            dense_linear_i8(model.buf1, output_raw, model.h1, model.output_dim,
                            model.q3, model.s3, model.b3);
        }
    } else {
        dense_relu_f(input_raw, model.buf1, model.input_dim, model.h1, model.w1, model.b1);
        if (model.h2 > 0) {
            dense_relu_f(model.buf1, model.buf2, model.h1, model.h2, model.w2, model.b2);
            dense_linear_f(model.buf2, output_raw, model.h2, model.output_dim, model.w3, model.b3);
        } else {
            dense_linear_f(model.buf1, output_raw, model.h1, model.output_dim, model.w3, model.b3);
        }
    }
}

void neural_physics_step(Player *p, InputState input, float dt) {
    (void)dt;
    if (!model_loaded) {
        physics_step(p, input, FIXED_DT);
        return;
    }

    float in[INPUT_DIM];
    float out[OUTPUT_DIM];
    pack_player_features(p, input, in);
    neural_forward(in, out);

    p->pos.x += out[0];
    p->pos.y += out[1];
    p->pos.z += out[2];
    p->vel.x = out[3];
    p->vel.y = out[4];
    p->vel.z = out[5];
    p->grounded = out[6] >= 0.5f ? 1 : 0;
}

double neural_forward_only_us(const float *input_raw, float *output_raw, int iterations) {
    if (!model_loaded || iterations <= 0) return 0.0;

#ifdef _WIN32
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    for (int i = 0; i < iterations; i++) neural_forward(input_raw, output_raw);
    QueryPerformanceCounter(&t1);
    return (double)(t1.QuadPart - t0.QuadPart) * 1000000.0 / (double)freq.QuadPart / (double)iterations;
#else
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++) neural_forward(input_raw, output_raw);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    return sec * 1000000.0 / (double)iterations;
#endif
}
