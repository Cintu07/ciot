// T2P vs T2 on identical weights: exactness, single-token matvec, and the 4-token SMMLA tile
// that speculative verification uses (plus a 64-token prompt batch).
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
#include "../kernels/t2p.h"
}

namespace {

float h2f(uint16_t h) { __fp16 f; std::memcpy(&f, &h, 2); return (float) f; }
uint16_t f2h(float v) { __fp16 f = (__fp16) v; uint16_t h; std::memcpy(&h, &f, 2); return h; }

struct Problem {
    int rows, n, cols;
    std::vector<ciot_block_t2x4> t2;
    std::vector<ciot_block_t2p> t2p;
    std::vector<ciot_block_q8_K> act;
    std::vector<ciot_block_q8_Kx4> act4;
};

Problem make(int rows, int n, int cols, std::mt19937& rng) {
    Problem p{rows, n, cols, {}, {}, {}, {}};
    const int nb = n / 128, nbk = n / 256;
    std::uniform_int_distribution<int> trit(-1, 1), q(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f), dy(0.001f, 0.1f);
    p.t2.resize((size_t) rows / 4 * nb);
    p.t2p.resize(p.t2.size());
    std::vector<int8_t> w(512);
    for (size_t b = 0; b < p.t2.size(); ++b) {
        for (auto& v : w) v = (int8_t) trit(rng);
        const int8_t* src[4] = {&w[0], &w[128], &w[256], &w[384]};
        float s4[4];
        for (auto& v : s4) v = h2f(f2h(sc(rng)));
        ciot_t2x4_pack(src, s4, &p.t2[b]);
        ciot_t2p_pack(src, s4, &p.t2p[b]);
    }
    p.act.resize((size_t) cols * nbk);
    for (auto& b : p.act) {
        b.d = dy(rng);
        for (auto& v : b.qs) v = (int8_t) q(rng);
        for (int j = 0; j < 16; ++j) { int s = 0; for (int k = 0; k < 16; ++k) s += b.qs[16 * j + k]; b.bsums[j] = (int16_t) s; }
    }
    p.act4.resize((size_t) cols / 4 * nbk);
    for (int cg = 0; cg < cols / 4; ++cg)
        for (int b = 0; b < nbk; ++b) {
            auto& o = p.act4[(size_t) cg * nbk + b];
            std::memset(&o, 0, sizeof(o));
            for (int r = 0; r < 4; ++r) {
                const auto& src = p.act[(size_t) (4 * cg + r) * nbk + b];
                o.d[r] = src.d;
                for (int c = 0; c < 32; ++c) std::memcpy(o.qs + 32 * c + 8 * r, src.qs + 8 * c, 8);
            }
        }
    return p;
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
    for (int rep = 0; rep < 7; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) once();
        runs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters);
    }
    quit.store(true);
    for (auto& th : pool) th.join();
    std::sort(runs.begin(), runs.end());
    return runs[3];
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::mt19937 rng(5);

    // Exactness against T2 (which is bit-identical to PrismML PQ2_0).
    {
        Problem p = make(64, 2048, 8, rng);
        const int nb = p.n / 128, nbk = p.n / 256;
        std::vector<int32_t> scratch(p.n / 32);
        std::vector<float> a((size_t) p.cols * p.rows), b(a.size()), c(a.size()), d(a.size());
        for (int col = 0; col < p.cols; ++col)
            for (int g = 0; g < p.rows / 4; ++g) {
                ciot_gemv_t2x4_q8_K_neon(p.n, &a[(size_t) col * p.rows + 4 * g], &p.t2[(size_t) g * nb], &p.act[(size_t) col * nbk]);
                ciot_gemv_t2p_q8_K(p.n, &b[(size_t) col * p.rows + 4 * g], &p.t2p[(size_t) g * nb], &p.act[(size_t) col * nbk]);
            }
        ciot_gemm_t2x4_q8_Kx4_i8mm(p.n, c.data(), p.rows, p.t2.data(), p.act4.data(), p.cols, p.rows, scratch.data());
        ciot_gemm_t2p_q8_Kx4(p.n, d.data(), p.rows, p.t2p.data(), p.act4.data(), p.cols, p.rows, scratch.data());
        int bad_gemv = 0, bad_gemm = 0, bad_cross = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            bad_gemv += a[i] != b[i];
            bad_gemm += c[i] != d[i];
            bad_cross += a[i] != d[i];
        }
        std::printf("bit-exact vs T2: gemv %s, gemm %s, gemm vs gemv %s (%zu outputs)\n", bad_gemv ? "FAIL" : "ok",
                    bad_gemm ? "FAIL" : "ok", bad_cross ? "FAIL" : "ok", a.size());
        if (bad_gemv || bad_gemm || bad_cross) return 1;
    }

    const int T = 8;
    // Single-token matvec, weights streamed from DRAM.
    {
        Problem p = make(32768, 8192, 4, rng);
        const int nb = p.n / 128, groups = p.rows / 4;
        std::vector<float> out(p.rows);
        const double bytes = (double) p.t2.size() * sizeof(ciot_block_t2x4);
        auto split = [&](int t, int nt) { return std::make_pair(groups * t / nt, groups * (t + 1) / nt); };
        const double t2 = time_pool(T, 10, [&](int t, int nt) { auto r = split(t, nt); for (int g = r.first; g < r.second; ++g) ciot_gemv_t2x4_q8_K_neon(p.n, &out[4 * g], &p.t2[(size_t) g * nb], p.act.data()); });
        const double tp = time_pool(T, 10, [&](int t, int nt) { auto r = split(t, nt); for (int g = r.first; g < r.second; ++g) ciot_gemv_t2p_q8_K(p.n, &out[4 * g], &p.t2p[(size_t) g * nb], p.act.data()); });
        std::printf("\nmatvec 32768x8192, 8 threads: T2 %.3f ms (%.1f GB/s) | T2P %.3f ms (%.1f GB/s)\n", t2 * 1e3, bytes / t2 / 1e9, tp * 1e3, bytes / tp / 1e9);
    }

    // SMMLA tiles: 4 tokens (speculative verify) and 64 tokens (prompt), MLP-sized weights.
    for (int cols : {4, 64}) {
        Problem p = make(17408, 5120, cols, rng);
        const int nb = p.n / 128, groups = p.rows / 4;
        std::vector<float> out((size_t) cols * p.rows);
        std::vector<std::vector<int32_t>> scratch(T, std::vector<int32_t>(p.n / 32));
        const double macs = (double) p.rows * p.n * cols;
        auto split = [&](int t, int nt) { return std::make_pair(groups * t / nt, groups * (t + 1) / nt); };
        const int iters = cols == 4 ? 20 : 3;
        const double t2 = time_pool(T, iters, [&](int t, int nt) {
            auto r = split(t, nt);
            ciot_gemm_t2x4_q8_Kx4_i8mm(p.n, out.data() + 4 * r.first, p.rows, &p.t2[(size_t) r.first * nb], p.act4.data(), cols, 4 * (r.second - r.first), scratch[t].data());
        });
        const double tp = time_pool(T, iters, [&](int t, int nt) {
            auto r = split(t, nt);
            ciot_gemm_t2p_q8_Kx4(p.n, out.data() + 4 * r.first, p.rows, &p.t2p[(size_t) r.first * nb], p.act4.data(), cols, 4 * (r.second - r.first), scratch[t].data());
        });
        std::printf("smmla %2d tokens x 17408x5120, 8 threads: T2 %.3f ms (%.0f GMAC/s) | T2P %.3f ms (%.0f GMAC/s) | %.2fx\n",
                    cols, t2 * 1e3, macs / t2 / 1e9, tp * 1e3, macs / tp / 1e9, t2 / tp);
    }
    return 0;
}
