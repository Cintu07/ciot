// Head-to-head of every ternary matvec kernel that can run Bonsai 2 on this CPU, on identical weights.
//
//   prism PTQ1_0 generic  what the 5.95 GB PTQ1_0 file runs on ARM today (arch-fallback.h aliases it)
//   prism PQ2_0  NEON     PrismML's ARM kernel for the 7.21 GB PQ2_0 file (ggml_vec_dot_pq2_0_q8_K)
//   ciot  PTQ1_0 NEON     kernels/ptq1_0_neon.c
//   ciot  T2x4   NEON     kernels/t2_neon.c (weights repacked at load)
//
// PrismML kernels are copied from PrismML-Eng/llama.cpp prism@6bfcd79.
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

extern "C" {
#include "../kernels/ptq1_0_neon.h"
#include "../kernels/t2_neon.h"
}

namespace {

float h2f(uint16_t h) { __fp16 f; std::memcpy(&f, &h, 2); return (float) f; }
uint16_t f2h(float v) { __fp16 f = (__fp16) v; uint16_t h; std::memcpy(&h, &f, 2); return h; }

struct block_pq2_0 { uint16_t d; uint8_t qs[32]; };
static_assert(sizeof(block_pq2_0) == 34, "pq2_0 block size");

// ---- PrismML kernels (verbatim logic) ---------------------------------------------------------

void prism_ptq1_0_q8_0_generic(int n, float* s, const void* vx, const void* vy) {
    const int nb = n / CIOT_QK_PTQ1_0;
    const ciot_block_ptq1_0* x = (const ciot_block_ptq1_0*) vx;
    const ciot_block_q8_0* y = (const ciot_block_q8_0*) vy;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    static const size_t stages[3] = {32, 16, 8};
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        int8_t q[CIOT_QK_PTQ1_0];
        int o = 0;
        size_t j = 0;
        for (size_t st = 0; st < 3; ++st) {
            const size_t c = stages[st];
            for (; j + c <= sizeof(x->qs); j += c)
                for (size_t nn = 0; nn < 5; ++nn)
                    for (size_t m = 0; m < c; ++m) {
                        const uint8_t v = x[i].qs[j + m] * pow3[nn];
                        q[o++] = (int8_t) ((((uint16_t) v * 3) >> 8) - 1);
                    }
        }
        for (size_t nn = 0; nn < 4; ++nn)
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                const uint8_t v = x[i].qh[h] * pow3[nn];
                q[o++] = (int8_t) ((((uint16_t) v * 3) >> 8) - 1);
            }
        const float d0 = h2f(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            const ciot_block_q8_0* yb = &y[i * 4 + k];
            int sb = 0;
            for (int b = 0; b < 32; ++b) sb += (int) q[k * 32 + b] * (int) yb->qs[b];
            sumi += h2f(yb->d) * sb;
        }
        sumf += d0 * sumi;
    }
    *s = sumf;
}

void prism_pq2_0_q8_K_neon(int n, float* s, const void* vx, const void* vy) {
    const block_pq2_0* x = (const block_pq2_0*) vx;
    const ciot_block_q8_K* y = (const ciot_block_q8_K*) vy;
    const int nb = n / 128;
    float sumf = 0.0f;
    static const uint8_t tbl_idx_lo[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    static const uint8_t tbl_idx_hi[16] = {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
    static const int8_t shift_vals[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
    const uint8x16_t idx_lo = vld1q_u8(tbl_idx_lo);
    const uint8x16_t idx_hi = vld1q_u8(tbl_idx_hi);
    const int8x16_t shifts = vld1q_s8(shift_vals);
    const uint8x16_t mask2 = vdupq_n_u8(0x03);
    const int8x16_t one = vdupq_n_s8(1);
    for (int i = 0; i < nb; i++) {
        const ciot_block_q8_K* yb = &y[i >> 1];
        const int8_t* q8 = yb->qs + 128 * (i & 1);
        int32x4_t acc = vdupq_n_s32(0);
        for (int k = 0; k < 4; k++) {
            const uint8x8_t raw = vld1_u8(&x[i].qs[8 * k]);
            const uint8x16_t raw16 = vcombine_u8(raw, raw);
            const int8x16_t qv0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(raw16, idx_lo), shifts), mask2)), one);
            const int8x16_t qv1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(raw16, idx_hi), shifts), mask2)), one);
            acc = vdotq_s32(acc, qv0, vld1q_s8(q8 + 32 * k));
            acc = vdotq_s32(acc, qv1, vld1q_s8(q8 + 32 * k + 16));
        }
        sumf += (h2f(x[i].d) * yb->d) * (float) vaddvq_s32(acc);
    }
    *s = sumf;
}

// ---- Packers ----------------------------------------------------------------------------------

void pack_ptq1_0(const int8_t* w, float scale, ciot_block_ptq1_0* b) {
    auto pack = [](int value) { return (uint8_t) ((value * 256 + 242) / 243); };
    for (int m = 0; m < 16; ++m) { int v = 0; for (int nn = 0; nn < 5; ++nn) v = v * 3 + (w[16 * nn + m] + 1); b->qs[m] = pack(v); }
    for (int m = 0; m < 8; ++m) { int v = 0; for (int nn = 0; nn < 5; ++nn) v = v * 3 + (w[80 + 8 * nn + m] + 1); b->qs[16 + m] = pack(v); }
    for (int h = 0; h < 2; ++h) { int v = 0; for (int nn = 0; nn < 4; ++nn) v = v * 3 + (w[120 + 2 * nn + h] + 1); b->qh[h] = pack(v * 3); }
    b->d = f2h(scale);
}

void pack_pq2_0(const int8_t* w, float scale, block_pq2_0* b) {
    std::memset(b->qs, 0, sizeof(b->qs));
    for (int k = 0; k < 4; ++k)
        for (int bb = 0; bb < 8; ++bb)
            for (int j = 0; j < 4; ++j) b->qs[8 * k + bb] |= (uint8_t) ((w[32 * k + 4 * bb + j] + 1) << (2 * j));
    b->d = f2h(scale);
}

// ---- One matrix in every format ---------------------------------------------------------------

struct Weights {
    int rows, cols;
    std::vector<int8_t> w;     // reference trits, row major
    std::vector<float> scale;  // per row per 128 block
    std::vector<ciot_block_ptq1_0> ptq1;
    std::vector<block_pq2_0> pq2;
    std::vector<ciot_block_t2x4> t2;
};

Weights make_weights(int rows, int cols, std::mt19937& rng, bool unit_scales) {
    Weights m{rows, cols, {}, {}, {}, {}, {}};
    const int nb = cols / 128;
    m.w.resize((size_t) rows * cols);
    m.scale.resize((size_t) rows * nb);
    std::uniform_int_distribution<int> trit(-1, 1);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (auto& v : m.w) v = (int8_t) trit(rng);
    for (auto& v : m.scale) v = unit_scales ? 1.0f : h2f(f2h(sc(rng)));
    m.ptq1.resize((size_t) rows * nb);
    m.pq2.resize((size_t) rows * nb);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < nb; ++i) {
            const int8_t* src = m.w.data() + (size_t) r * cols + 128 * i;
            pack_ptq1_0(src, m.scale[(size_t) r * nb + i], &m.ptq1[(size_t) r * nb + i]);
            pack_pq2_0(src, m.scale[(size_t) r * nb + i], &m.pq2[(size_t) r * nb + i]);
        }
    m.t2.resize((size_t) rows / 4 * nb);
    for (int r4 = 0; r4 < rows / 4; ++r4)
        for (int i = 0; i < nb; ++i) {
            const int8_t* src[4];
            float sc4[4];
            for (int r = 0; r < 4; ++r) {
                src[r] = m.w.data() + (size_t) (4 * r4 + r) * cols + 128 * i;
                sc4[r] = m.scale[(size_t) (4 * r4 + r) * nb + i];
            }
            ciot_t2x4_pack(src, sc4, &m.t2[(size_t) r4 * nb + i]);
        }
    return m;
}

struct Acts {
    std::vector<int8_t> q;
    std::vector<ciot_block_q8_0> q8_0;
    std::vector<ciot_block_q8_K> q8_K;
};

// Same int8 values in both activation formats; Q8_0 scales are all equal to the Q8_K scale.
Acts make_acts(int cols, std::mt19937& rng, bool unit_scales) {
    Acts a;
    a.q.resize(cols);
    std::uniform_int_distribution<int> qd(-127, 127);
    for (auto& v : a.q) v = (int8_t) qd(rng);
    const float d = unit_scales ? 1.0f : h2f(f2h(0.0123f));
    a.q8_0.resize(cols / 32);
    for (int b = 0; b < cols / 32; ++b) { a.q8_0[b].d = f2h(d); std::memcpy(a.q8_0[b].qs, &a.q[32 * b], 32); }
    a.q8_K.resize(cols / 256);
    for (int b = 0; b < cols / 256; ++b) {
        a.q8_K[b].d = d;
        std::memcpy(a.q8_K[b].qs, &a.q[256 * b], 256);
        for (int j = 0; j < 16; ++j) { int sum = 0; for (int k = 0; k < 16; ++k) sum += a.q[256 * b + 16 * j + k]; a.q8_K[b].bsums[j] = (int16_t) sum; }
    }
    return a;
}

enum Kernel { PRISM_PTQ1, PRISM_PQ2, CIOT_PTQ1, CIOT_T2 };
const char* kName[] = {"prism PTQ1_0 generic", "prism PQ2_0  NEON   ", "ciot  PTQ1_0 NEON   ", "ciot  T2x4   NEON   "};
const double kBitsPerWeight[] = {1.75, 2.125, 1.75, 2.125};

void run_rows(Kernel k, const Weights& m, const Acts& a, float* out, int r0, int r1) {
    const int nb = m.cols / 128;
    switch (k) {
    case PRISM_PTQ1: for (int r = r0; r < r1; ++r) prism_ptq1_0_q8_0_generic(m.cols, out + r, &m.ptq1[(size_t) r * nb], a.q8_0.data()); break;
    case PRISM_PQ2:  for (int r = r0; r < r1; ++r) prism_pq2_0_q8_K_neon(m.cols, out + r, &m.pq2[(size_t) r * nb], a.q8_K.data()); break;
    case CIOT_PTQ1:  for (int r = r0; r < r1; ++r) ciot_vec_dot_ptq1_0_q8_0_neon(m.cols, out + r, &m.ptq1[(size_t) r * nb], a.q8_0.data()); break;
    case CIOT_T2:    for (int r = r0; r < r1; r += 4) ciot_gemv_t2x4_q8_K_neon(m.cols, out + r, &m.t2[(size_t) r / 4 * nb], a.q8_K.data()); break;
    }
}

double time_kernel(Kernel k, const Weights& m, const Acts& a, float* out, int threads, int iters) {
    std::vector<double> runs;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t) {
                const int r0 = (int) ((int64_t) m.rows / 4 * t / threads) * 4, r1 = (int) ((int64_t) m.rows / 4 * (t + 1) / threads) * 4;
                pool.emplace_back(run_rows, k, std::cref(m), std::cref(a), out, r0, r1);
            }
            for (auto& th : pool) th.join();
        }
        runs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters);
    }
    std::sort(runs.begin(), runs.end());
    return runs[2];
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::mt19937 rng(7);

    // Exactness: with unit scales every kernel must return the integer dot product exactly.
    for (int trial = 0; trial < 40; ++trial) {
        const int rows = 8, cols = 256 * (1 + trial % 12);
        Weights m = make_weights(rows, cols, rng, true);
        Acts a = make_acts(cols, rng, true);
        for (int k = 0; k < 4; ++k) {
            std::vector<float> out(rows);
            run_rows((Kernel) k, m, a, out.data(), 0, rows);
            for (int r = 0; r < rows; ++r) {
                long ref = 0;
                for (int c = 0; c < cols; ++c) ref += (long) m.w[(size_t) r * cols + c] * a.q[c];
                if ((long) out[r] != ref) { std::printf("FAIL %s row %d cols %d: %g vs %ld\n", kName[k], r, cols, out[r], ref); return 1; }
            }
        }
    }
    std::printf("exactness: all 4 kernels match the integer reference on 40 shapes\n");

    // Scaled agreement against PrismML PQ2_0 (the official ARM path).
    {
        Weights m = make_weights(64, 5120, rng, false);
        Acts a = make_acts(5120, rng, false);
        std::vector<float> ref(64), out(64);
        run_rows(PRISM_PQ2, m, a, ref.data(), 0, 64);
        for (int k = 0; k < 4; ++k) {
            run_rows((Kernel) k, m, a, out.data(), 0, 64);
            double worst = 0, mag = 0;
            for (int r = 0; r < 64; ++r) { worst = std::max(worst, (double) std::fabs(out[r] - ref[r])); mag = std::max(mag, (double) std::fabs(ref[r])); }
            std::printf("  %s max |diff| vs prism PQ2_0 = %.2e (output magnitude %.2e)\n", kName[k], worst, mag);
        }
    }

    // Speed on a matrix that does not fit in cache.
    const int rows = 32768, cols = 8192;
    Weights m = make_weights(rows, cols, rng, false);
    Acts a = make_acts(cols, rng, false);
    std::vector<float> out(rows);
    const double weights = (double) rows * cols;
    const double weights_per_token_27b = 25.6e9; // Bonsai 2 27B backbone + LM head
    std::printf("\n%dx%d matvec, weights streamed from DRAM. Last column: compute-limited tok/s for Bonsai 2 27B.\n", rows, cols);
    for (int threads : {1, 4, 8}) {
        std::printf("threads=%d\n", threads);
        double base = 0;
        for (int k = 0; k < 4; ++k) {
            const double t = time_kernel((Kernel) k, m, a, out.data(), threads, threads == 1 ? 3 : 10);
            if (k == 0) base = t;
            const double gw = weights / t / 1e9;
            std::printf("  %s %8.3f ms  %6.1f Gweights/s  %5.1f GB/s  %4.1fx vs today  ~%4.1f tok/s\n", kName[k], t * 1e3, gw,
                        gw * kBitsPerWeight[k] / 8, base / t, gw * 1e9 / weights_per_token_27b);
        }
    }
    return 0;
}
