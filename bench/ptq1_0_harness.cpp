// Correctness and speed of ciot_vec_dot_ptq1_0_q8_0_neon against PrismML's generic kernel.
//
// The generic kernel below is copied from PrismML-Eng/llama.cpp (prism branch, 6bfcd79,
// ggml/src/ggml-cpu/quants.c), which is what Snapdragon runs today: arch-fallback.h aliases
// PTQ1_0 on ARM to the generic loop.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

extern "C" {
#include "../kernels/ptq1_0_neon.h"
}

namespace {

float h2f(uint16_t h) { __fp16 f; std::memcpy(&f, &h, 2); return (float) f; }
uint16_t f2h(float v) { __fp16 f = (__fp16) v; uint16_t h; std::memcpy(&h, &f, 2); return h; }

void vec_dot_ptq1_0_q8_0_generic(int n, float* s, const void* vx, const void* vy) {
    const int qk = CIOT_QK_PTQ1_0;
    const int nb = n / qk;
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
            for (; j + c <= sizeof(x->qs); j += c) {
                for (size_t nn = 0; nn < 5; ++nn) {
                    for (size_t m = 0; m < c; ++m) {
                        const uint8_t v = x[i].qs[j + m] * pow3[nn];
                        const int16_t xi = ((uint16_t) v * 3) >> 8;
                        q[o++] = (int8_t) (xi - 1);
                    }
                }
            }
        }
        for (size_t nn = 0; nn < 4; ++nn) {
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                const uint8_t v = x[i].qh[h] * pow3[nn];
                const int16_t xi = ((uint16_t) v * 3) >> 8;
                q[o++] = (int8_t) (xi - 1);
            }
        }
        const float d0 = h2f(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            const ciot_block_q8_0* yb = &y[i * 4 + k];
            const float d1 = h2f(yb->d);
            int sumi_block = 0;
            for (int b = 0; b < 32; ++b) sumi_block += (int) q[k * 32 + b] * (int) yb->qs[b];
            sumi += d1 * sumi_block;
        }
        sumf += d0 * sumi;
    }
    *s = sumf;
}

// Packs 128 weights in {-1,0,1} the way quantize_row_tq1_0_ref does (TQ1_0 and PTQ1_0 share it).
void pack_block(const int8_t* w, float scale, ciot_block_ptq1_0* b) {
    auto pack = [](int value) { return (uint8_t) ((value * 256 + 242) / 243); };
    for (int m = 0; m < 16; ++m) {
        int v = 0;
        for (int nn = 0; nn < 5; ++nn) v = v * 3 + (w[16 * nn + m] + 1);
        b->qs[m] = pack(v);
    }
    for (int m = 0; m < 8; ++m) {
        int v = 0;
        for (int nn = 0; nn < 5; ++nn) v = v * 3 + (w[80 + 8 * nn + m] + 1);
        b->qs[16 + m] = pack(v);
    }
    for (int h = 0; h < 2; ++h) {
        int v = 0;
        for (int nn = 0; nn < 4; ++nn) v = v * 3 + (w[120 + 2 * nn + h] + 1);
        b->qh[h] = pack(v * 3);
    }
    b->d = f2h(scale);
}

struct Matrix {
    int rows, cols, nb;
    std::vector<ciot_block_ptq1_0> w;
};

Matrix make_matrix(int rows, int cols, std::mt19937& rng) {
    Matrix m{rows, cols, cols / CIOT_QK_PTQ1_0, {}};
    m.w.resize((size_t) rows * m.nb);
    std::uniform_int_distribution<int> trit(-1, 1);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    int8_t w[CIOT_QK_PTQ1_0];
    for (auto& b : m.w) {
        for (auto& v : w) v = (int8_t) trit(rng);
        pack_block(w, sc(rng), &b);
    }
    return m;
}

std::vector<ciot_block_q8_0> make_activations(int cols, std::mt19937& rng) {
    std::vector<ciot_block_q8_0> y(cols / CIOT_QK8_0);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> sc(0.001f, 0.1f);
    for (auto& b : y) {
        b.d = f2h(sc(rng));
        for (auto& v : b.qs) v = (int8_t) q(rng);
    }
    return y;
}

using DotFn = void (*)(int, float*, const void*, const void*);

void matvec(DotFn dot, const Matrix& m, const ciot_block_q8_0* y, float* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r) dot(m.cols, out + r, m.w.data() + (size_t) r * m.nb, y);
}

double time_matvec(DotFn dot, const Matrix& m, const ciot_block_q8_0* y, float* out, int threads, int iters) {
    std::vector<double> runs;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t) {
                const int r0 = (int) ((int64_t) m.rows * t / threads), r1 = (int) ((int64_t) m.rows * (t + 1) / threads);
                pool.emplace_back(matvec, dot, std::cref(m), y, out, r0, r1);
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
    std::mt19937 rng(42);

    // 1. The packer must agree with the generic decoder, or the comparison below means nothing.
    {
        int8_t w[CIOT_QK_PTQ1_0];
        std::uniform_int_distribution<int> trit(-1, 1);
        ciot_block_ptq1_0 b;
        ciot_block_q8_0 one[4];
        for (int trial = 0; trial < 2000; ++trial) {
            for (auto& v : w) v = (int8_t) trit(rng);
            pack_block(w, 1.0f, &b);
            for (int pos = 0; pos < CIOT_QK_PTQ1_0; ++pos) {
                std::memset(one, 0, sizeof(one));
                for (auto& blk : one) blk.d = f2h(1.0f);
                one[pos / 32].qs[pos % 32] = 1; // picks out weight `pos`
                float got;
                vec_dot_ptq1_0_q8_0_generic(CIOT_QK_PTQ1_0, &got, &b, one);
                if ((int) got != w[pos]) { std::printf("FAIL packer: weight %d expected %d got %g\n", pos, w[pos], got); return 1; }
            }
        }
        std::printf("packer round-trip vs generic decoder: ok (2000 blocks x 128 weights)\n");
    }

    // 2. NEON vs generic on random rows, including the exact integer case (all scales 1).
    {
        double worst = 0;
        for (int trial = 0; trial < 200; ++trial) {
            const int cols = CIOT_QK_PTQ1_0 * (1 + trial % 40);
            Matrix m = make_matrix(1, cols, rng);
            auto y = make_activations(cols, rng);
            if (trial % 2 == 0) { for (auto& b : m.w) b.d = f2h(1.0f); for (auto& b : y) b.d = f2h(1.0f); }
            float g, n;
            vec_dot_ptq1_0_q8_0_generic(cols, &g, m.w.data(), y.data());
            ciot_vec_dot_ptq1_0_q8_0_neon(cols, &n, m.w.data(), y.data());
            if (trial % 2 == 0 && g != n) { std::printf("FAIL exact: cols %d generic %.1f neon %.1f\n", cols, g, n); return 1; }
            double scale = 0;
            for (auto& b : y) scale += 127.0 * std::fabs(h2f(b.d)) * 32;
            worst = std::max(worst, std::fabs((double) g - n) / (scale * 0.05));
        }
        std::printf("neon vs generic: exact on integer inputs, worst scaled float diff %.2e (fp32 rounding order only)\n", worst);
    }

    // 3. Speed. 5120x5120 stays mostly in cache; 32768x8192 (~59 MB of weights) streams from DRAM.
    const int shapes[2][2] = {{5120, 5120}, {32768, 8192}};
    const unsigned hw = std::thread::hardware_concurrency();
    for (auto& sh : shapes) {
        Matrix m = make_matrix(sh[0], sh[1], rng);
        auto y = make_activations(sh[1], rng);
        std::vector<float> out_g(sh[0]), out_n(sh[0]);
        const double weights = (double) sh[0] * sh[1];
        const double bytes = (double) m.w.size() * sizeof(ciot_block_ptq1_0);
        std::printf("\n%dx%d (%.1f MB of PTQ1_0 weights)\n", sh[0], sh[1], bytes / 1e6);
        for (int threads : {1, 4, (int) hw}) {
            const int iters = threads == 1 ? 3 : 10;
            const double tg = time_matvec(vec_dot_ptq1_0_q8_0_generic, m, y.data(), out_g.data(), threads, iters);
            const double tn = time_matvec(ciot_vec_dot_ptq1_0_q8_0_neon, m, y.data(), out_n.data(), threads, iters);
            std::printf("  threads=%d  generic %8.3f ms (%6.1f Gweights/s, %5.1f GB/s) | neon %8.3f ms (%6.1f Gweights/s, %5.1f GB/s) | %.1fx\n",
                        threads, tg * 1e3, weights / tg / 1e9, bytes / tg / 1e9, tn * 1e3, weights / tn / 1e9, bytes / tn / 1e9, tg / tn);
        }
    }
    return 0;
}
