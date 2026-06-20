#include "neural_simd.h"
#include <math.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

static int g_has_avx2 = -1;
static int g_has_vnni = -1;
static const char *g_kernel = "scalar";

void neural_simd_init(void) {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    if (g_has_avx2 < 0) {
        g_has_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
        g_has_vnni = __builtin_cpu_supports("avxvnni") ? 1 : 0;
        if (!g_has_vnni && __builtin_cpu_supports("avx512vnni"))
            g_has_vnni = 1;
    }
#else
    g_has_avx2 = 0;
    g_has_vnni = 0;
#endif
    if (g_has_vnni)
        g_kernel = "avx2";
    else if (g_has_avx2)
        g_kernel = "avx2";
    else
        g_kernel = "scalar";
}

const char *nn_simd_kernel_name(void) {
    if (g_has_avx2 < 0) neural_simd_init();
    return g_kernel;
}

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)

static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline float vmax8_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

static inline int hsum256_epi32(__m256i v) {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

static float dot_fp32_avx2(int n, const float *a, const float *b) {
    __m256 vacc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        vacc = _mm256_fmadd_ps(va, vb, vacc);
#else
        vacc = _mm256_add_ps(vacc, _mm256_mul_ps(va, vb));
#endif
    }
    float sum = hsum256_ps(vacc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float dot_int8_avx2(int n, const int8_t *w, float scale, const float *x) {
    __m256 vacc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32_t pack = 0;
        memcpy(&pack, w + i, 4);
        __m128i w4 = _mm_cvtsi32_si128(pack);
        __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w4));
        __m256 xf = _mm256_loadu_ps(x + i);
#if defined(__FMA__)
        vacc = _mm256_fmadd_ps(wf, xf, vacc);
#else
        vacc = _mm256_add_ps(vacc, _mm256_mul_ps(wf, xf));
#endif
    }
    float sum = hsum256_ps(vacc);
    for (; i < n; i++) sum += (float)w[i] * x[i];
    return sum * scale;
}

#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)

static void quantize_u8_relu(const float *x, int n, uint8_t *out, float *ascale) {
    float amax = 0.0f;
    int i = 0;
    __m256 vmax = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, v);
    }
    amax = vmax8_ps(vmax);
    for (; i < n; i++)
        if (x[i] > amax) amax = x[i];

    if (amax < 1e-8f) {
        *ascale = 1.0f;
        memset(out, 0, (size_t)n);
        return;
    }
    *ascale = amax / 255.0f;
    float inv = 255.0f / amax;
    for (i = 0; i < n; i++)
        out[i] = (uint8_t)(x[i] * inv + 0.5f);
}

static float dot_vnni_row(int n, const int8_t *w, float wscale,
                          const uint8_t *xq, float xscale) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + i));
        __m256i wv = _mm256_loadu_si256((const __m256i *)(w + i));
#if defined(__AVXVNNI__)
        acc = _mm256_dpbusd_avx_epi32(acc, xv, wv);
#else
        acc = _mm256_dpbusd_epi32(acc, xv, wv);
#endif
    }
    int32_t dots = hsum256_epi32(acc);
    for (; i < n; i++)
        dots += (int32_t)xq[i] * (int32_t)w[i];
    return (float)dots * wscale * xscale;
}

static void dense_int8_vnni(const float *in, float *out, int in_n, int out_n,
                            const int8_t *w, const float *scales, const float *b,
                            int use_relu) {
    uint8_t xq[768];
    float xscale;
    if (in_n > (int)(sizeof(xq) / sizeof(xq[0]))) {
        /* fallback shouldn't happen for our MLP sizes */
        for (int o = 0; o < out_n; o++)
            out[o] = b[o];
        return;
    }
    quantize_u8_relu(in, in_n, xq, &xscale);
    for (int o = 0; o < out_n; o++) {
        const int8_t *row = w + (size_t)o * (size_t)in_n;
        float sum = dot_vnni_row(in_n, row, scales[o], xq, xscale) + b[o];
        out[o] = use_relu ? (sum > 0.0f ? sum : 0.0f) : sum;
    }
}

#endif /* VNNI */

static void dense_int8_avx2(const float *in, float *out, int in_n, int out_n,
                            const int8_t *w, const float *scales, const float *b,
                            int use_relu) {
    for (int o = 0; o < out_n; o++) {
        const int8_t *row = w + (size_t)o * (size_t)in_n;
        float sum = dot_int8_avx2(in_n, row, scales[o], in) + b[o];
        out[o] = use_relu ? (sum > 0.0f ? sum : 0.0f) : sum;
    }
}

static void dense_fp32_avx2(const float *in, float *out, int in_n, int out_n,
                            const float *w, const float *b, int use_relu) {
    for (int o = 0; o < out_n; o++) {
        const float *row = w + (size_t)o * (size_t)in_n;
        float sum = dot_fp32_avx2(in_n, row, in) + b[o];
        out[o] = use_relu ? (sum > 0.0f ? sum : 0.0f) : sum;
    }
}

#endif /* AVX2 */

/* --- Scalar fallbacks --- */

static float dot_fp32_scalar(int n, const float *a, const float *b) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float dot_int8_scalar(int n, const int8_t *w, float scale, const float *x) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += (float)w[i] * x[i];
    return sum * scale;
}

static void dense_int8_scalar(const float *in, float *out, int in_n, int out_n,
                              const int8_t *w, const float *scales, const float *b,
                              int use_relu) {
    for (int o = 0; o < out_n; o++) {
        const int8_t *row = w + (size_t)o * (size_t)in_n;
        float sum = dot_int8_scalar(in_n, row, scales[o], in) + b[o];
        out[o] = use_relu ? (sum > 0.0f ? sum : 0.0f) : sum;
    }
}

static void dense_fp32_scalar(const float *in, float *out, int in_n, int out_n,
                              const float *w, const float *b, int use_relu) {
    for (int o = 0; o < out_n; o++) {
        const float *row = w + (size_t)o * (size_t)in_n;
        float sum = dot_fp32_scalar(in_n, row, in) + b[o];
        out[o] = use_relu ? (sum > 0.0f ? sum : 0.0f) : sum;
    }
}

/* --- Public dispatch --- */

float nn_dot_fp32(int n, const float *a, const float *b) {
    if (g_has_avx2 < 0) neural_simd_init();
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    if (g_has_avx2) return dot_fp32_avx2(n, a, b);
#endif
    return dot_fp32_scalar(n, a, b);
}

float nn_dot_int8_row(int n, const int8_t *w, float scale, const float *x) {
    if (g_has_avx2 < 0) neural_simd_init();
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    if (g_has_avx2) return dot_int8_avx2(n, w, scale, x);
#endif
    return dot_int8_scalar(n, w, scale, x);
}

void nn_dense_int8_row_vnni(const float *in, float *out, int in_n, int out_n,
                            const int8_t *w, const float *scales, const float *b,
                            int use_relu) {
    /* W8A32: AVX2 int8 weights x float activations (VNNI W8A8 hurt rollout fidelity). */
    (void)use_relu;
    nn_dense_int8_row_simd(in, out, in_n, out_n, w, scales, b, use_relu);
}

void nn_dense_int8_row_simd(const float *in, float *out, int in_n, int out_n,
                            const int8_t *w, const float *scales, const float *b,
                            int use_relu) {
    /* Scalar W8A32: AVX2 int8->float promote was slower than scalar on these MLP widths. */
    (void)in_n;
    if (g_has_avx2 < 0) neural_simd_init();
    dense_int8_scalar(in, out, in_n, out_n, w, scales, b, use_relu);
}

void nn_dense_fp32_simd(const float *in, float *out, int in_n, int out_n,
                        const float *w, const float *b, int use_relu) {
    if (g_has_avx2 < 0) neural_simd_init();
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    if (g_has_avx2) {
        dense_fp32_avx2(in, out, in_n, out_n, w, b, use_relu);
        return;
    }
#endif
    dense_fp32_scalar(in, out, in_n, out_n, w, b, use_relu);
}
