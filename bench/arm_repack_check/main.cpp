// Direct CPU_REPACK tests. Kept outside upstream's tests directory.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "repack.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool check_case(ggml_backend_t backend, ggml_backend_buffer_type_t repack_buffer,
                       int k, int rows, int columns, bool supported, int pattern = 0) {
    ggml_context * weights = ggml_init({1024 * 1024, nullptr, true});
    ggml_context * ops = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor * w = ggml_new_tensor_2d(weights, GGML_TYPE_PQ2_0, k, rows);
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors_from_buft(weights, repack_buffer);
    if (!wb) { std::fprintf(stderr, "weight allocation failed\n"); std::exit(1); }

    bool ok = (w->extra != nullptr) == supported;
    if (!supported || !ok) {
        std::printf("shape k=%d rows=%d: repack %s, expected %s\n", k, rows,
                    w->extra ? "selected" : "declined", supported ? "selected" : "declined");
        ggml_backend_buffer_free(wb);
        ggml_free(ops);
        ggml_free(weights);
        return ok;
    }

    std::vector<block_pq2_0> packed(rows * k / QK_PQ2_0);
    std::vector<int> trits(rows * k);
    std::vector<float> scales(rows);
    for (int r = 0; r < rows; ++r) {
        scales[r] = (r % 3 + 1) * 0.125f;
        for (int b = 0; b < k / QK_PQ2_0; ++b) {
            block_pq2_0 & block = packed[r * (k / QK_PQ2_0) + b];
            block.d = ggml_fp32_to_fp16(scales[r]);
            for (int j = 0; j < QK_PQ2_0; ++j) {
                const int index = b * QK_PQ2_0 + j;
                const int code = pattern == 1 ? 2 : pattern == 2 ? 1 : pattern == 3 ? 0 :
                                 (index * 17 + r * 13 + index / 7) % 3;
                trits[r * k + index] = code - 1;
                block.qs[j / 4] |= uint8_t(code << (2 * (j % 4)));
            }
        }
    }
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size() * sizeof(block_pq2_0));

    // Integers with absmax 127 make Q8_K conversion exact; dy is +/-1.
    std::vector<float> activations(k * columns);
    for (int c = 0; c < columns; ++c) {
        for (int j = 0; j < k; ++j) {
            activations[c * k + j] = float((j * 37 + c * 19 + j / 11) % 255 - 127);
        }
        if (pattern == 4 || pattern == 5) {
            for (int j = 0; j < k; ++j) {
                activations[c * k + j] = pattern == 4 ? 0.0f : (c % 2 ? -127.0f : 127.0f);
            }
        } else {
            activations[c * k] = 127.0f;
            activations[c * k + 1] = -127.0f;
        }
    }
    ggml_tensor * a = ggml_new_tensor_2d(ops, GGML_TYPE_F32, k, columns);
    ggml_tensor * product = ggml_mul_mat(ops, w, a);
    ggml_cgraph * graph = ggml_new_graph_custom(ops, 64, false);
    ggml_build_forward_expand(graph, product);
    ggml_backend_buffer_t ob = ggml_backend_alloc_ctx_tensors(ops, backend);
    if (!ob) { std::fprintf(stderr, "operator allocation failed\n"); std::exit(1); }
    ggml_backend_tensor_set(a, activations.data(), 0, activations.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        std::exit(1);
    }
    std::vector<float> actual(rows * columns);
    ggml_backend_tensor_get(product, actual.data(), 0, actual.size() * sizeof(float));
    for (int c = 0; c < columns; ++c) {
        for (int r = 0; r < rows; ++r) {
            int sum = 0;
            for (int j = 0; j < k; ++j) {
                sum += trits[r * k + j] * int(activations[c * k + j]);
            }
            const float expected = float(sum) * scales[r];
            if (actual[c * rows + r] != expected) {
                std::fprintf(stderr, "k=%d rows=%d columns=%d row=%d col=%d got=%g expected=%g\n",
                             k, rows, columns, r, c, actual[c * rows + r], expected);
                ok = false;
            }
        }
    }
    std::printf("k=%d rows=%d columns=%d pattern=%d: %s\n", k, rows, columns, pattern, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(ob);
    ggml_backend_buffer_free(wb);
    ggml_free(ops);
    ggml_free(weights);
    return ok;
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    if (!ggml_cpu_has_neon() || !ggml_cpu_has_dotprod()) {
        std::puts("SKIP: ARM NEON + dotprod required; no ARM execution was validated.");
        ggml_backend_free(backend);
        return 2;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
    ggml_backend_buffer_type_t repack_buffer = nullptr;
    if (get_extra) {
        for (auto buffers = get_extra(dev); buffers && *buffers; ++buffers) {
            if (std::strcmp(ggml_backend_buft_name(*buffers), "CPU_REPACK") == 0) {
                repack_buffer = *buffers;
                break;
            }
        }
    }
    if (!repack_buffer) { std::fprintf(stderr, "CPU_REPACK buffer unavailable\n"); return 1; }
    int total = 0;
    int passed = 0;
    for (int k : {256, 512, 1024}) {
        for (int rows : {4, 8, 12}) {
            for (int columns : {1, 2, 3, 4, 5, 7, 8}) {
                passed += check_case(backend, repack_buffer, k, rows, columns, true);
                ++total;
            }
        }
    }
    // The selector must decline invalid row tiles and non-Q8_K widths.
    for (int rows : {1, 3, 5, 7}) {
        passed += check_case(backend, repack_buffer, 256, rows, 1, false);
        ++total;
    }
    passed += check_case(backend, repack_buffer, 128, 4, 1, false);
    ++total;
    for (int pattern : {1, 2, 3, 4, 5}) {
        for (int columns : {3, 5}) {
            passed += check_case(backend, repack_buffer, 256, 8, columns, true, pattern);
            ++total;
        }
    }
    std::printf("%d/%d direct repacked cases passed\n", passed, total);
    ggml_backend_free(backend);
    return passed == total ? 0 : 1;
}
