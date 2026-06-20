#include "neural.h"
#include "observation.h"
#include "neural_simd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define NNLIKELY(x)   __builtin_expect(!!(x), 1)
#define NNUNLIKELY(x) __builtin_expect(!!(x), 0)
#define NNPREFETCH(p) __builtin_prefetch((p), 0, 3)
#else
#define NNLIKELY(x)   (x)
#define NNUNLIKELY(x) (x)
#define NNPREFETCH(p) ((void)0)
#endif

typedef struct {
    int input_dim, h1, h2, output_dim;
    int quant; /* 0 = FP32 runtime */
    float *input_mean;
    float *input_std;
    float *output_mean;
    float *output_std;
    float *w1, *b1;
    float *w2, *b2;
    float *w3, *b3;
    int8_t *w1_q, *w2_q, *w3_q;
    uint8_t *w1_q4, *w2_q4, *w3_q4;
    float *w1_scale, *w2_scale, *w3_scale;
    float w1_layer_scale, w2_layer_scale, w3_layer_scale;
    float *buf1;
    float *buf2;
    float *buf_out;
} MLPModel;

static MLPModel model;
static int model_loaded = 0;
static int model_quant_type = 0;

static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) return *(float *)&sign;
        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FFu;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        return *(float *)&bits;
    }
    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    return *(float *)&bits;
}

static inline int8_t w4_at(const uint8_t *pack, int idx) {
    uint8_t byte = pack[idx >> 1];
    int v = (idx & 1) ? (byte >> 4) : (byte & 0x0Fu);
    return (int8_t)(v > 7 ? v - 16 : v);
}

static inline float dot_unrolled(int n, const float * restrict a, const float * restrict b) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static inline float dot_int8_row(int n, const int8_t * restrict w, float scale, const float * restrict x) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += (float)w[i]     * x[i];
        s1 += (float)w[i + 1] * x[i + 1];
        s2 += (float)w[i + 2] * x[i + 2];
        s3 += (float)w[i + 3] * x[i + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) sum += (float)w[i] * x[i];
    return sum * scale;
}

static inline float dot_int4_row(int n, const uint8_t * restrict row, float scale, const float * restrict x) {
    float sum = 0.0f;
    int i = 0;
    for (; i + 1 < n; i += 2) {
        uint8_t byte = row[i >> 1];
        int8_t w0 = (int8_t)(byte & 0x0Fu);
        int8_t w1 = (int8_t)(byte >> 4);
        if (w0 > 7) w0 = (int8_t)(w0 - 16);
        if (w1 > 7) w1 = (int8_t)(w1 - 16);
        sum += (float)w0 * x[i] + (float)w1 * x[i + 1];
    }
    if (i < n) {
        int8_t w0 = w4_at(row, i);
        sum += (float)w0 * x[i];
    }
    return sum * scale;
}

static void dense(const float * restrict in, float * restrict out,
                  int in_n, int out_n,
                  const float * restrict w, const float * restrict b,
                  int use_relu) {
    nn_dense_fp32_simd(in, out, in_n, out_n, w, b, use_relu);
}

static void dense_int8_row(const float * restrict in, float * restrict out,
                           int in_n, int out_n,
                           const int8_t * restrict w, const float * restrict scales,
                           const float * restrict b, int use_relu) {
    nn_dense_int8_row_vnni(in, out, in_n, out_n, w, scales, b, use_relu);
}

static void dense_int8_layer(const float * restrict in, float * restrict out,
                             int in_n, int out_n,
                             const int8_t * restrict w, float scale,
                             const float * restrict b, int use_relu) {
    for (int o = 0; o < out_n; o++) {
        const int8_t *row = w + (size_t)o * (size_t)in_n;
        float sum = nn_dot_int8_row(in_n, row, scale, in) + b[o];
        out[o] = use_relu ? relu_f(sum) : sum;
    }
}

static void dense_int4_row(const float * restrict in, float * restrict out,
                           int in_n, int out_n,
                           const uint8_t * restrict pack, const float * restrict scales,
                           const float * restrict b, int use_relu) {
    const size_t row_bytes = ((size_t)in_n + 1) / 2;
    for (int o = 0; o < out_n; o++) {
        const uint8_t *row = pack + (size_t)o * row_bytes;
        if (o + 1 < out_n) NNPREFETCH(pack + (size_t)(o + 1) * row_bytes);
        float sum = dot_int4_row(in_n, row, scales[o], in) + b[o];
        out[o] = use_relu ? relu_f(sum) : sum;
    }
}

static void dense_layer1_norm(const float * restrict in_raw, float * restrict out) {
    const int in_n = model.input_dim;
    const int out_n = model.h1;
    const float * restrict mean = model.input_mean;
    const float * restrict std = model.input_std;

    if (model.quant == QUANT_INT8_ROW) {
        for (int o = 0; o < out_n; o++) {
            const int8_t *row = model.w1_q + (size_t)o * (size_t)in_n;
            float scale = model.w1_scale[o];
            float sum = model.b1[o];
            for (int i = 0; i < in_n; i++)
                sum += (float)row[i] * scale * ((in_raw[i] - mean[i]) / std[i]);
            out[o] = relu_f(sum);
        }
        return;
    }

    if (model.quant == QUANT_INT8_LAYER) {
        float scale = model.w1_layer_scale;
        for (int o = 0; o < out_n; o++) {
            const int8_t *row = model.w1_q + (size_t)o * (size_t)in_n;
            float sum = model.b1[o];
            for (int i = 0; i < in_n; i++)
                sum += (float)row[i] * scale * ((in_raw[i] - mean[i]) / std[i]);
            out[o] = relu_f(sum);
        }
        return;
    }

    if (model.quant == QUANT_INT4_ROW) {
        const size_t row_bytes = ((size_t)in_n + 1) / 2;
        for (int o = 0; o < out_n; o++) {
            const uint8_t *row = model.w1_q4 + (size_t)o * row_bytes;
            float scale = model.w1_scale[o];
            float sum = model.b1[o];
            for (int i = 0; i < in_n; i++)
                sum += (float)w4_at(row, i) * scale * ((in_raw[i] - mean[i]) / std[i]);
            out[o] = relu_f(sum);
        }
        return;
    }

    const float * restrict w = model.w1;
    const float * restrict b = model.b1;
    float *xnorm = model.buf2;
    for (int i = 0; i < in_n; i++)
        xnorm[i] = (in_raw[i] - mean[i]) / std[i];
    for (int o = 0; o < out_n; o++) {
        const float *row = w + (size_t)o * (size_t)in_n;
        if (o + 1 < out_n) NNPREFETCH(w + (size_t)(o + 1) * (size_t)in_n);
        float sum = nn_dot_fp32(in_n, row, xnorm) + b[o];
        out[o] = relu_f(sum);
    }
}

static void denorm_output(float * restrict output_raw) {
    const int n = model.output_dim;
    const float * restrict mean = model.output_mean;
    const float * restrict std = model.output_std;
    const float * restrict src = model.buf_out;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        output_raw[i]     = src[i]     * std[i]     + mean[i];
        output_raw[i + 1] = src[i + 1] * std[i + 1] + mean[i + 1];
        output_raw[i + 2] = src[i + 2] * std[i + 2] + mean[i + 2];
        output_raw[i + 3] = src[i + 3] * std[i + 3] + mean[i + 3];
    }
    for (; i < n; i++)
        output_raw[i] = src[i] * std[i] + mean[i];
}

static void free_model(void) {
    free(model.input_mean);
    free(model.input_std);
    free(model.output_mean);
    free(model.output_std);
    free(model.w1);
    free(model.b1);
    free(model.w2);
    free(model.b2);
    free(model.w3);
    free(model.b3);
    free(model.w1_q);
    free(model.w2_q);
    free(model.w3_q);
    free(model.w1_q4);
    free(model.w2_q4);
    free(model.w3_q4);
    free(model.w1_scale);
    free(model.w2_scale);
    free(model.w3_scale);
    free(model.buf1);
    free(model.buf2);
    free(model.buf_out);
    memset(&model, 0, sizeof(model));
    model_loaded = 0;
    model_quant_type = 0;
}

static int read_floats(FILE *f, float *dst, int n) {
    return (int)fread(dst, sizeof(float), (size_t)n, f) == (size_t)n;
}

static int read_bytes(FILE *f, void *dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

static int alloc_common_buffers(void) {
    model.buf1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    int scratch_n = model.input_dim;
    if (model.h1 > scratch_n) scratch_n = model.h1;
    if (model.h2 > scratch_n) scratch_n = model.h2;
    model.buf2 = (float *)malloc((size_t)scratch_n * sizeof(float));
    model.buf_out = (float *)malloc((size_t)model.output_dim * sizeof(float));
    return model.buf1 && model.buf2 && model.buf_out;
}

static int load_norm_stats(FILE *f) {
    model.input_mean = (float *)calloc((size_t)model.input_dim, sizeof(float));
    model.input_std = (float *)calloc((size_t)model.input_dim, sizeof(float));
    model.output_mean = (float *)calloc((size_t)model.output_dim, sizeof(float));
    model.output_std = (float *)calloc((size_t)model.output_dim, sizeof(float));
    if (!model.input_mean || !model.input_std || !model.output_mean || !model.output_std)
        return 0;
    if (!read_floats(f, model.input_mean, model.input_dim)) return 0;
    if (!read_floats(f, model.input_std, model.input_dim)) return 0;
    if (!read_floats(f, model.output_mean, model.output_dim)) return 0;
    if (!read_floats(f, model.output_std, model.output_dim)) return 0;
    for (int i = 0; i < model.input_dim; i++)
        if (model.input_std[i] < 1e-6f) model.input_std[i] = 1.0f;
    for (int i = 0; i < model.output_dim; i++)
        if (model.output_std[i] < 1e-6f) model.output_std[i] = 1.0f;
    return 1;
}

static int load_fp32_weights(FILE *f) {
    model.w1 = (float *)malloc((size_t)model.h1 * (size_t)model.input_dim * sizeof(float));
    model.b1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    model.w2 = (float *)malloc((size_t)model.h2 * (size_t)model.h1 * sizeof(float));
    model.b2 = (float *)malloc((size_t)model.h2 * sizeof(float));
    model.w3 = (float *)malloc((size_t)model.output_dim * (size_t)model.h2 * sizeof(float));
    model.b3 = (float *)malloc((size_t)model.output_dim * sizeof(float));
    if (!model.w1 || !model.b1 || !model.w2 || !model.b2 || !model.w3 || !model.b3)
        return 0;
    if (!read_floats(f, model.w1, model.h1 * model.input_dim)) return 0;
    if (!read_floats(f, model.b1, model.h1)) return 0;
    if (!read_floats(f, model.w2, model.h2 * model.h1)) return 0;
    if (!read_floats(f, model.b2, model.h2)) return 0;
    if (!read_floats(f, model.w3, model.output_dim * model.h2)) return 0;
    if (!read_floats(f, model.b3, model.output_dim)) return 0;
    model.quant = 0;
    return 1;
}

static int load_int8_row_layer(FILE *f, int out_n, int in_n,
                               int8_t **wq, float **scales, float **bias) {
    *scales = (float *)malloc((size_t)out_n * sizeof(float));
    *wq = (int8_t *)malloc((size_t)out_n * (size_t)in_n);
    *bias = (float *)malloc((size_t)out_n * sizeof(float));
    if (!*scales || !*wq || !*bias) return 0;
    if (!read_floats(f, *scales, out_n)) return 0;
    if (!read_bytes(f, *wq, (size_t)out_n * (size_t)in_n)) return 0;
    if (!read_floats(f, *bias, out_n)) return 0;
    return 1;
}

static int load_int8_layer_layer(FILE *f, int out_n, int in_n,
                                 int8_t **wq, float *scale, float **bias) {
    *wq = (int8_t *)malloc((size_t)out_n * (size_t)in_n);
    *bias = (float *)malloc((size_t)out_n * sizeof(float));
    if (!*wq || !*bias) return 0;
    if (!read_floats(f, scale, 1)) return 0;
    if (!read_bytes(f, *wq, (size_t)out_n * (size_t)in_n)) return 0;
    if (!read_floats(f, *bias, out_n)) return 0;
    return 1;
}

static int load_int4_row_layer(FILE *f, int out_n, int in_n,
                               uint8_t **wq4, float **scales, float **bias) {
    size_t row_bytes = ((size_t)in_n + 1) / 2;
    size_t packed = (size_t)out_n * row_bytes;
    *scales = (float *)malloc((size_t)out_n * sizeof(float));
    *wq4 = (uint8_t *)malloc(packed);
    *bias = (float *)malloc((size_t)out_n * sizeof(float));
    if (!*scales || !*wq4 || !*bias) return 0;
    if (!read_floats(f, *scales, out_n)) return 0;
    if (!read_bytes(f, *wq4, packed)) return 0;
    if (!read_floats(f, *bias, out_n)) return 0;
    return 1;
}

static int load_fp16_layer(FILE *f, int out_n, int in_n, float **w, float **bias) {
    size_t n = (size_t)out_n * (size_t)in_n;
    uint16_t *half = (uint16_t *)malloc(n * sizeof(uint16_t));
    *w = (float *)malloc(n * sizeof(float));
    *bias = (float *)malloc((size_t)out_n * sizeof(float));
    if (!half || !*w || !*bias) {
        free(half);
        return 0;
    }
    if (!read_bytes(f, half, n * sizeof(uint16_t))) {
        free(half);
        return 0;
    }
    for (size_t i = 0; i < n; i++)
        (*w)[i] = half_to_float(half[i]);
    free(half);
    if (!read_floats(f, *bias, out_n)) return 0;
    return 1;
}

static int load_quant_v2(FILE *f, uint32_t quant_type) {
    model.quant = (int)quant_type;
    model_quant_type = (int)quant_type;

    if (quant_type == QUANT_INT8_ROW) {
        if (!load_int8_row_layer(f, model.h1, model.input_dim, &model.w1_q, &model.w1_scale, &model.b1)) return 0;
        if (!load_int8_row_layer(f, model.h2, model.h1, &model.w2_q, &model.w2_scale, &model.b2)) return 0;
        if (!load_int8_row_layer(f, model.output_dim, model.h2, &model.w3_q, &model.w3_scale, &model.b3)) return 0;
        return 1;
    }

    if (quant_type == QUANT_INT8_LAYER) {
        if (!load_int8_layer_layer(f, model.h1, model.input_dim, &model.w1_q, &model.w1_layer_scale, &model.b1)) return 0;
        if (!load_int8_layer_layer(f, model.h2, model.h1, &model.w2_q, &model.w2_layer_scale, &model.b2)) return 0;
        if (!load_int8_layer_layer(f, model.output_dim, model.h2, &model.w3_q, &model.w3_layer_scale, &model.b3)) return 0;
        return 1;
    }

    if (quant_type == QUANT_INT4_ROW) {
        if (!load_int4_row_layer(f, model.h1, model.input_dim, &model.w1_q4, &model.w1_scale, &model.b1)) return 0;
        if (!load_int4_row_layer(f, model.h2, model.h1, &model.w2_q4, &model.w2_scale, &model.b2)) return 0;
        if (!load_int4_row_layer(f, model.output_dim, model.h2, &model.w3_q4, &model.w3_scale, &model.b3)) return 0;
        return 1;
    }

    if (quant_type == QUANT_FP16) {
        if (!load_fp16_layer(f, model.h1, model.input_dim, &model.w1, &model.b1)) return 0;
        if (!load_fp16_layer(f, model.h2, model.h1, &model.w2, &model.b2)) return 0;
        if (!load_fp16_layer(f, model.output_dim, model.h2, &model.w3, &model.b3)) return 0;
        model.quant = 0; /* expanded to FP32 at load */
        model_quant_type = (int)QUANT_FP16;
        return 1;
    }

    fprintf(stderr, "Error: unsupported quant type %u\n", quant_type);
    return 0;
}

int neural_load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: model file not found: %s\n", path);
        return 0;
    }

    free_model();

    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || magic != MODEL_MAGIC) {
        fprintf(stderr, "Error: invalid model magic in %s\n", path);
        fclose(f);
        return 0;
    }
    if (fread(&version, 4, 1, f) != 1 ||
        (version != MODEL_VERSION_FP32 && version != MODEL_VERSION_QUANT)) {
        fprintf(stderr, "Error: unsupported model version in %s\n", path);
        fclose(f);
        return 0;
    }

    uint32_t quant_type = 0;
    if (version == MODEL_VERSION_QUANT) {
        if (fread(&quant_type, 4, 1, f) != 1) {
            fprintf(stderr, "Error: truncated quant header in %s\n", path);
            fclose(f);
            return 0;
        }
    }

    int dims[4];
    if (fread(dims, sizeof(int), 4, f) != 4) {
        fprintf(stderr, "Error: truncated model header in %s\n", path);
        fclose(f);
        return 0;
    }

    model.input_dim = dims[0];
    model.h1 = dims[1];
    model.h2 = dims[2];
    model.output_dim = dims[3];

    int ok = load_norm_stats(f);
    ok = ok && alloc_common_buffers();

    if (version == MODEL_VERSION_FP32)
        ok = ok && load_fp32_weights(f);
    else
        ok = ok && load_quant_v2(f, quant_type);

    fclose(f);
    if (!ok) {
        fprintf(stderr, "Error: truncated model weights in %s\n", path);
        free_model();
        return 0;
    }

    obs_set_patch_n(patch_from_input_dim(model.input_dim));
    neural_simd_init();
    model_loaded = 1;
    return 1;
}

int neural_require_model(const char *path) {
    if (!neural_load_model(path)) {
        fprintf(stderr, "Fatal: required model could not be loaded: %s\n", path);
        exit(1);
    }
    return 1;
}

int neural_model_loaded(void) { return model_loaded; }

int neural_quant_type(void) { return model_quant_type; }

const char *neural_quant_name(void) {
    switch (model_quant_type) {
    case 0: return "fp32";
    case QUANT_INT8_ROW: return "int8_row";
    case QUANT_INT8_LAYER: return "int8_layer";
    case QUANT_INT4_ROW: return "int4_row";
    case QUANT_FP16: return "fp16";
    default: return "unknown";
    }
}

const char *neural_kernel_name(void) {
    return nn_simd_kernel_name();
}

void neural_forward(const float *input_raw, float *output_raw) {
    if (model.quant == QUANT_INT8_ROW) {
        dense_layer1_norm(input_raw, model.buf1);
        dense_int8_row(model.buf1, model.buf2, model.h1, model.h2,
                       model.w2_q, model.w2_scale, model.b2, 1);
        dense_int8_row(model.buf2, model.buf_out, model.h2, model.output_dim,
                       model.w3_q, model.w3_scale, model.b3, 0);
    } else if (model.quant == QUANT_INT8_LAYER) {
        dense_layer1_norm(input_raw, model.buf1);
        dense_int8_layer(model.buf1, model.buf2, model.h1, model.h2,
                         model.w2_q, model.w2_layer_scale, model.b2, 1);
        dense_int8_layer(model.buf2, model.buf_out, model.h2, model.output_dim,
                         model.w3_q, model.w3_layer_scale, model.b3, 0);
    } else if (model.quant == QUANT_INT4_ROW) {
        dense_layer1_norm(input_raw, model.buf1);
        dense_int4_row(model.buf1, model.buf2, model.h1, model.h2,
                       model.w2_q4, model.w2_scale, model.b2, 1);
        dense_int4_row(model.buf2, model.buf_out, model.h2, model.output_dim,
                       model.w3_q4, model.w3_scale, model.b3, 0);
    } else {
        dense_layer1_norm(input_raw, model.buf1);
        dense(model.buf1, model.buf2, model.h1, model.h2, model.w2, model.b2, 1);
        dense(model.buf2, model.buf_out, model.h2, model.output_dim, model.w3, model.b3, 0);
    }
    denorm_output(output_raw);
}

void neural_physics_step(Player *p, InputState input, float dt) {
    (void)dt;
    if (NNUNLIKELY(!model_loaded)) {
        fprintf(stderr, "Fatal: neural_physics_step called without a loaded model\n");
        exit(1);
    }

    Observation obs;
    build_observation(p, input, &obs);

    float in[INPUT_DIM_MAX];
    pack_observation(&obs, in);

    float out[OUTPUT_DIM];
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
