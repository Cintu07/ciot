// Replays the ternary matvecs of one Bonsai 2 27B decode step with the T2 kernel, using the
// model's real shapes, to separate kernel cost from runtime cost.
//
// A quarter of the model (12 linear-attention blocks + 4 full-attention blocks, ~1.7 GB of
// weights) is enough to stream from DRAM; times are scaled x4 to a full token.
//
// Variants:
//   ggml   dynamic chunks, 4 per thread, barrier after every matvec (what the runtime does)
//   static one contiguous slice per thread, barrier after every matvec
//   fused  static, plus gate+up as one matvec and attention q+k+v as one matvec
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

extern "C" {
#include "../kernels/t2_neon.h"
}

namespace {

struct Mat {
    int rows, n;
    std::vector<ciot_block_t2x4> w;
};

Mat make(int rows, int n) {
    Mat m{rows, n, {}};
    m.w.resize((size_t) rows / 4 * (n / 128));
    // Content does not matter for timing; any byte pattern is a valid T2 block.
    std::memset(m.w.data(), 0x5A, m.w.size() * sizeof(ciot_block_t2x4));
    for (auto& b : m.w)
        for (auto& d : b.d) d = 0x2000; // small positive fp16
    return m;
}

struct Op {
    const Mat* m;
};

struct Spin {
    std::atomic<int> count{0}, phase{0};
    int n;
    explicit Spin(int n_) : n(n_) {}
    void wait() {
        const int p = phase.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            phase.fetch_add(1, std::memory_order_acq_rel);
        } else {
            while (phase.load(std::memory_order_acquire) == p) {}
        }
    }
};

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int H = 5120, FF = 17408, NTH = 8;

    // Linear-attention block: attn_qkv, attn_gate, ssm_out, ffn_gate, ffn_up, ffn_down.
    // Full-attention block:   attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down.
    std::vector<Mat> mats;
    mats.reserve(200);
    std::vector<std::vector<int>> seq_plain, seq_fused; // indices into mats, per op
    auto add = [&](int rows, int n) { mats.push_back(make(rows, n)); return (int) mats.size() - 1; };
    for (int b = 0; b < 16; ++b) {
        const bool full = (b % 4 == 3);
        if (!full) {
            int qkv = add(10240, H), gate = add(6144, H), out = add(H, 6144);
            int fg = add(FF, H), fu = add(FF, H), fd = add(H, FF);
            int fgu = add(2 * FF, H);
            seq_plain.insert(seq_plain.end(), {{qkv}, {gate}, {out}, {fg}, {fu}, {fd}});
            seq_fused.insert(seq_fused.end(), {{qkv}, {gate}, {out}, {fgu}, {fd}});
        } else {
            int q = add(12288, H), k = add(1024, H), v = add(1024, H), o = add(H, 6144);
            int fg = add(FF, H), fu = add(FF, H), fd = add(H, FF);
            int qkv = add(12288 + 2048, H), fgu = add(2 * FF, H);
            seq_plain.insert(seq_plain.end(), {{q}, {k}, {v}, {o}, {fg}, {fu}, {fd}});
            seq_fused.insert(seq_fused.end(), {{qkv}, {o}, {fgu}, {fd}});
        }
    }

    double plain_bytes = 0;
    for (auto& op : seq_plain) plain_bytes += (double) mats[op[0]].w.size() * sizeof(ciot_block_t2x4);
    std::printf("quarter model: %zu matvecs, %.2f GB of T2 weights streamed per pass\n", seq_plain.size(), plain_bytes / 1e9);

    std::vector<ciot_block_q8_K> act(FF / 256);
    for (auto& b : act) { b.d = 0.01f; std::memset(b.qs, 3, sizeof(b.qs)); std::memset(b.bsums, 0, sizeof(b.bsums)); }
    std::vector<float> out(2 * FF + 64);

    enum Mode { GGML, STATIC, FUSED };
    const char* names[] = {"ggml   (dynamic chunks)", "static (one slice/thread)", "fused  (gate+up, q+k+v)"};

    for (int mode = 0; mode < 3; ++mode) {
        const auto& seq = mode == FUSED ? seq_fused : seq_plain;
        Spin barrier(NTH);
        std::atomic<int> chunk{0};
        std::atomic<bool> go{false}, quit{false};
        std::atomic<int> started{0};

        auto pass = [&](int t) {
            for (auto& op : seq) {
                const Mat& m = mats[op[0]];
                const int nb = m.n / 128, groups = m.rows / 4;
                if (mode == GGML) {
                    const int nchunk = NTH * 4, per = (groups + nchunk - 1) / nchunk;
                    for (int c = t; c < nchunk; c = chunk.fetch_add(1) + NTH) {
                        for (int g = c * per; g < std::min(groups, (c + 1) * per); ++g)
                            ciot_gemv_t2x4_q8_K_neon(m.n, out.data() + 4 * g, &m.w[(size_t) g * nb], act.data());
                    }
                    barrier.wait();
                    if (t == 0) chunk.store(0);
                    barrier.wait();
                } else {
                    const int g0 = groups * t / NTH, g1 = groups * (t + 1) / NTH;
                    for (int g = g0; g < g1; ++g) ciot_gemv_t2x4_q8_K_neon(m.n, out.data() + 4 * g, &m.w[(size_t) g * nb], act.data());
                    barrier.wait();
                }
            }
        };

        std::vector<std::thread> pool;
        Spin start(NTH);
        for (int t = 1; t < NTH; ++t)
            pool.emplace_back([&, t] {
                for (;;) {
                    start.wait();
                    if (quit.load()) return;
                    pass(t);
                }
            });

        std::vector<double> runs;
        for (int rep = 0; rep < 8; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            start.wait();
            pass(0);
            runs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        quit.store(true);
        start.wait();
        for (auto& th : pool) th.join();
        std::sort(runs.begin(), runs.end());
        const double ms = runs[runs.size() / 2] * 1e3;
        std::printf("  %-26s %7.2f ms per quarter pass -> %6.1f ms per token, %6.1f GB/s, %5.1f tok/s matmul-only\n",
                    names[mode], ms, ms * 4, plain_bytes / (ms / 1e3) / 1e9, 1e3 / (ms * 4));
    }
    return 0;
}
