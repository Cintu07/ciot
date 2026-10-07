// Prompt-processing kernel check: T2 x Q8_Kx4 with SMMLA vs running the T2 decode kernel once per
// activation column (what the runtime patch did before this kernel).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
#include <vector>

extern "C" {
#include "../kernels/t2_neon.h"
}

namespace {

float h2f(uint16_t h) { __fp16 f; std::memcpy(&f, &h, 2); return (float) f; }
uint16_t f2h(float v) { __fp16 f = (__fp16) v; uint16_t h; std::memcpy(&h, &f, 2); return h; }

struct Problem {
    int rows, n, cols;
    std::vector<ciot_block_t2x4> w;         // rows/4 x n/128
    std::vector<ciot_block_q8_K> act;       // cols x n/256, plain rows (for the decode kernel)
    std::vector<ciot_block_q8_Kx4> act4;    // cols/4 x n/256, interleaved (for SMMLA)
};

Problem make_problem(int rows, int n, int cols, std::mt19937& rng, bool unit) {
    Problem p{rows, n, cols, {}, {}, {}};
    const int nb = n / 128, nbk = n / 256;
    std::uniform_int_distribution<int> trit(-1, 1), q(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f), dy(0.001f, 0.1f);
    p.w.resize((size_t) rows / 4 * nb);
    std::vector<int8_t> w(4 * 128);
    for (auto& b : p.w) {
        for (auto& v : w) v = (int8_t) trit(rng);
        const int8_t* src[4] = {&w[0], &w[128], &w[256], &w[384]};
        float s4[4];
        for (auto& v : s4) v = unit ? 1.0f : h2f(f2h(sc(rng)));
        ciot_t2x4_pack(src, s4, &b);
    }
    p.act.resize((size_t) cols * nbk);
    for (auto& b : p.act) {
        b.d = unit ? 1.0f : dy(rng);
        for (auto& v : b.qs) v = (int8_t) q(rng);
        for (int j = 0; j < 16; ++j) { int s = 0; for (int k = 0; k < 16; ++k) s += b.qs[16 * j + k]; b.bsums[j] = (int16_t) s; }
    }
    p.act4.resize((size_t) cols / 4 * nbk);
    for (int cg = 0; cg < cols / 4; ++cg)
        for (int b = 0; b < nbk; ++b) {
            ciot_block_q8_Kx4& o = p.act4[(size_t) cg * nbk + b];
            std::memset(&o, 0, sizeof(o));
            for (int r = 0; r < 4; ++r) {
                const ciot_block_q8_K& src = p.act[(size_t) (4 * cg + r) * nbk + b];
                o.d[r] = src.d;
                for (int c = 0; c < 32; ++c) std::memcpy(o.qs + 32 * c + 8 * r, src.qs + 8 * c, 8);
            }
        }
    return p;
}

// out[col * rows + row]; rows [r0, r1) only.
void run_gemv(const Problem& p, float* out, int r0, int r1) {
    const int nb = p.n / 128, nbk = p.n / 256;
    for (int c = 0; c < p.cols; ++c)
        for (int r = r0; r < r1; r += 4)
            ciot_gemv_t2x4_q8_K_neon(p.n, out + (size_t) c * p.rows + r, &p.w[(size_t) r / 4 * nb], &p.act[(size_t) c * nbk]);
}

void run_i8mm(const Problem& p, float* out, int r0, int r1, int32_t* scratch) {
    const int nb = p.n / 128;
    ciot_gemm_t2x4_q8_Kx4_i8mm(p.n, out + r0, p.rows, &p.w[(size_t) r0 / 4 * nb], p.act4.data(), p.cols, r1 - r0, scratch);
}

double time_pool(int threads, int iters, const std::function<void(int, int)>& work) {
    std::atomic<int> generation{0}, done{0};
    std::atomic<bool> quit{false};
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t)
        pool.emplace_back([&, t] {
            int seen = 0;
            while (true) {
                int g;
                while ((g = generation.load(std::memory_order_acquire)) == seen)
                    if (quit.load(std::memory_order_relaxed)) return;
                seen = g;
                work(t, threads);
                done.fetch_add(1, std::memory_order_acq_rel);
            }
        });
    auto once = [&] {
        done.store(0);
        generation.fetch_add(1, std::memory_order_acq_rel);
        work(0, threads);
        while (done.load(std::memory_order_acquire) != threads - 1) {}
    };
    once();
    std::vector<double> runs;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) once();
        runs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters);
    }
    quit.store(true);
    for (auto& th : pool) th.join();
    std::sort(runs.begin(), runs.end());
    return runs[2];
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::mt19937 rng(11);

    for (int unit = 1; unit >= 0; --unit) {
        Problem p = make_problem(64, 1024, 12, rng, unit);
        std::vector<float> ref((size_t) p.cols * p.rows), got((size_t) p.cols * p.rows);
        std::vector<int32_t> scratch(p.n / 32);
        run_gemv(p, ref.data(), 0, p.rows);
        run_i8mm(p, got.data(), 0, p.rows, scratch.data());
        double worst = 0, mag = 0;
        for (size_t i = 0; i < ref.size(); ++i) { worst = std::max(worst, (double) std::fabs(ref[i] - got[i])); mag = std::max(mag, (double) std::fabs(ref[i])); }
        if (unit && worst != 0) { std::printf("FAIL: i8mm differs from decode kernel on integer inputs (max diff %g)\n", worst); return 1; }
        std::printf("%s inputs: max |i8mm - decode kernel| = %.2e (magnitude %.2e)\n", unit ? "integer" : "scaled", worst, mag);
    }

    // Shape of a Bonsai 2 27B MLP down-projection-sized matmul (rows x n), 64 prompt tokens.
    const int rows = 5120, n = 17408, cols = 64;
    Problem p = make_problem(rows, n, cols, rng, false);
    std::vector<float> out((size_t) cols * rows);
    const double macs = (double) rows * n * cols;
    std::printf("\n%dx%d weights x %d tokens\n", rows, n, cols);
    for (int threads : {1, 8}) {
        std::vector<std::vector<int32_t>> scratch(threads, std::vector<int32_t>(n / 32));
        auto split = [&](int t, int nt) {
            const int r0 = rows / 4 * t / nt * 4, r1 = rows / 4 * (t + 1) / nt * 4;
            return std::make_pair(r0, r1);
        };
        const double tg = time_pool(threads, 2, [&](int t, int nt) { auto r = split(t, nt); run_gemv(p, out.data(), r.first, r.second); });
        const double ti = time_pool(threads, 2, [&](int t, int nt) { auto r = split(t, nt); run_i8mm(p, out.data(), r.first, r.second, scratch[t].data()); });
        std::printf("  threads=%d  decode kernel per token %8.2f ms (%6.1f GMAC/s, ~%5.1f prompt tok/s on 27B) | i8mm %8.2f ms (%6.1f GMAC/s, ~%5.1f prompt tok/s) | %.2fx\n",
                    threads, tg * 1e3, macs / tg / 1e9, macs / tg / 25.6e9, ti * 1e3, macs / ti / 1e9, macs / ti / 25.6e9, tg / ti);
    }
    return 0;
}
