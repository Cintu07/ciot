// Streaming read bandwidth: T threads each sum their slice of a 1 GiB buffer with NEON loads.
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t bytes = 1ull << 30, n = bytes / 16;
    uint8_t* buf = (uint8_t*)_aligned_malloc(bytes, 64);
    std::memset(buf, 1, bytes);
    unsigned hw = std::thread::hardware_concurrency();
    std::printf("hardware threads: %u\n", hw);
    for (unsigned T : {1u, 2u, 4u, 8u, hw}) {
        if (T > hw) continue;
        std::vector<double> runs;
        for (int rep = 0; rep < 5; ++rep) {
            std::vector<std::thread> th; std::vector<uint32_t> out(T * 16);
            auto t0 = std::chrono::steady_clock::now();
            for (unsigned t = 0; t < T; ++t) th.emplace_back([&, t] {
                const uint8x16_t* p = (const uint8x16_t*)buf + n / T * t;
                uint32x4_t a0 = vdupq_n_u32(0), a1 = a0, a2 = a0, a3 = a0;
                for (size_t i = 0; i < n / T; i += 4) {
                    a0 = vpadalq_u16(a0, vpaddlq_u8(p[i])); a1 = vpadalq_u16(a1, vpaddlq_u8(p[i+1]));
                    a2 = vpadalq_u16(a2, vpaddlq_u8(p[i+2])); a3 = vpadalq_u16(a3, vpaddlq_u8(p[i+3]));
                }
                out[t * 16] = vaddvq_u32(vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3)));
            });
            for (auto& x : th) x.join();
            runs.push_back(bytes / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9);
        }
        std::sort(runs.begin(), runs.end());
        std::printf("threads=%2u  read bandwidth best %.1f GB/s  median %.1f GB/s\n", T, runs[4], runs[2]);
    }
}
