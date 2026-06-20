#include "neural.h"
#include "observation.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct {
    int input_dim, h1, h2, output_dim;
    float *input_mean;
    float *input_std;
    float *output_mean;
    float *output_std;
    float *w1, *b1;
    float *w2, *b2;
    float *w3, *b3;
    float *buf1;
    float *buf2;
    float *buf_out;
} MLPModel;

static MLPModel model;
static int model_loaded = 0;

static float relu(float x) { return x > 0.0f ? x : 0.0f; }

static void dense(const float *in, float *out, int in_n, int out_n, const float *w, const float *b, int use_relu) {
    for (int o = 0; o < out_n; o++) {
        float sum = b[o];
        const float *row = w + o * in_n;
        for (int i = 0; i < in_n; i++) sum += row[i] * in[i];
        out[o] = use_relu ? relu(sum) : sum;
    }
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
    free(model.buf1);
    free(model.buf2);
    free(model.buf_out);
    memset(&model, 0, sizeof(model));
    model_loaded = 0;
}

static int read_floats(FILE *f, float *dst, int n) {
    return (int)fread(dst, sizeof(float), (size_t)n, f) == (size_t)n;
}

int neural_load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: model file not found: %s\n", path);
        return 0;
    }

    free_model();

    uint32_t magic;
    uint32_t version;
    if (fread(&magic, 4, 1, f) != 1 || magic != MODEL_MAGIC) {
        fprintf(stderr, "Error: invalid model magic in %s\n", path);
        fclose(f);
        return 0;
    }
    if (fread(&version, 4, 1, f) != 1 || version != 1) {
        fprintf(stderr, "Error: unsupported model version in %s\n", path);
        fclose(f);
        return 0;
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

    model.input_mean = (float *)calloc((size_t)model.input_dim, sizeof(float));
    model.input_std = (float *)calloc((size_t)model.input_dim, sizeof(float));
    model.output_mean = (float *)calloc((size_t)model.output_dim, sizeof(float));
    model.output_std = (float *)calloc((size_t)model.output_dim, sizeof(float));
    model.w1 = (float *)malloc((size_t)model.h1 * (size_t)model.input_dim * sizeof(float));
    model.b1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    model.w2 = (float *)malloc((size_t)model.h2 * (size_t)model.h1 * sizeof(float));
    model.b2 = (float *)malloc((size_t)model.h2 * sizeof(float));
    model.w3 = (float *)malloc((size_t)model.output_dim * (size_t)model.h2 * sizeof(float));
    model.b3 = (float *)malloc((size_t)model.output_dim * sizeof(float));
    model.buf1 = (float *)malloc((size_t)model.h1 * sizeof(float));
    model.buf2 = (float *)malloc((size_t)model.h2 * sizeof(float));
    model.buf_out = (float *)malloc((size_t)model.output_dim * sizeof(float));

    int ok = model.input_mean && model.input_std && model.output_mean && model.output_std &&
             model.w1 && model.b1 && model.w2 && model.b2 && model.w3 && model.b3 &&
             model.buf1 && model.buf2 && model.buf_out;

    ok = ok && read_floats(f, model.input_mean, model.input_dim);
    ok = ok && read_floats(f, model.input_std, model.input_dim);
    ok = ok && read_floats(f, model.output_mean, model.output_dim);
    ok = ok && read_floats(f, model.output_std, model.output_dim);
    ok = ok && read_floats(f, model.w1, model.h1 * model.input_dim);
    ok = ok && read_floats(f, model.b1, model.h1);
    ok = ok && read_floats(f, model.w2, model.h2 * model.h1);
    ok = ok && read_floats(f, model.b2, model.h2);
    ok = ok && read_floats(f, model.w3, model.output_dim * model.h2);
    ok = ok && read_floats(f, model.b3, model.output_dim);

    fclose(f);
    if (!ok) {
        fprintf(stderr, "Error: truncated model weights in %s\n", path);
        free_model();
        return 0;
    }

    for (int i = 0; i < model.input_dim; i++)
        if (model.input_std[i] < 1e-6f) model.input_std[i] = 1.0f;
    for (int i = 0; i < model.output_dim; i++)
        if (model.output_std[i] < 1e-6f) model.output_std[i] = 1.0f;

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

void neural_forward(const float *input_raw, float *output_raw) {
    float norm_in[INPUT_DIM];
    for (int i = 0; i < model.input_dim; i++)
        norm_in[i] = (input_raw[i] - model.input_mean[i]) / model.input_std[i];

    dense(norm_in, model.buf1, model.input_dim, model.h1, model.w1, model.b1, 1);
    dense(model.buf1, model.buf2, model.h1, model.h2, model.w2, model.b2, 1);
    dense(model.buf2, model.buf_out, model.h2, model.output_dim, model.w3, model.b3, 0);

    for (int i = 0; i < model.output_dim; i++)
        output_raw[i] = model.buf_out[i] * model.output_std[i] + model.output_mean[i];
}

void neural_physics_step(Player *p, InputState input, float dt) {
    (void)dt;
    if (!model_loaded) {
        fprintf(stderr, "Fatal: neural_physics_step called without a loaded model\n");
        exit(1);
    }

    Observation obs;
    build_observation(p, input, &obs);

    float in[INPUT_DIM];
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
