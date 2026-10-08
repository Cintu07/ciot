#include "../../include/Ciot.h"

#include <algorithm>
#include <cmath>

namespace ciot {

void rmsnorm(float* out, const float* x, const float* weight, std::uint32_t n, float eps) {
    if (!out || !x || !weight || n == 0) {
        return;
    }

    float ss = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        ss += x[i] * x[i];
    }

    const float inv_rms = 1.0f / std::sqrt((ss / static_cast<float>(n)) + eps);
    for (std::uint32_t i = 0; i < n; ++i) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

void rope_pairwise(float* x, std::uint32_t dim, std::uint32_t position, float theta) {
    if (!x || dim < 2 || theta <= 0.0f) {
        return;
    }

    const std::uint32_t even_dim = dim & ~1U;
    for (std::uint32_t i = 0; i < even_dim; i += 2U) {
        const float freq = std::pow(theta, -static_cast<float>(i) / static_cast<float>(even_dim));
        const float angle = static_cast<float>(position) * freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float a = x[i];
        const float b = x[i + 1U];
        x[i] = (a * c) - (b * s);
        x[i + 1U] = (a * s) + (b * c);
    }
}

bool rope_table_init(RopeTable* table, std::uint32_t max_position, std::uint32_t dim, float theta) {
    if (!table || max_position == 0 || dim < 2 || theta <= 0.0f) {
        return false;
    }

    table->max_position = 0;
    table->dim = 0;
    table->cos = nullptr;
    table->sin = nullptr;
    const std::uint32_t even_dim = dim & ~1U;
    const std::size_t values = static_cast<std::size_t>(max_position) * even_dim;
    table->cos = static_cast<float*>(aligned_malloc(values * sizeof(float)));
    table->sin = static_cast<float*>(aligned_malloc(values * sizeof(float)));
    if (!table->cos || !table->sin) {
        rope_table_free(table);
        return false;
    }

    table->max_position = max_position;
    table->dim = even_dim;

    for (std::uint32_t pos = 0; pos < max_position; ++pos) {
        const std::size_t base = static_cast<std::size_t>(pos) * even_dim;
        for (std::uint32_t i = 0; i < even_dim; i += 2U) {
            const float freq = std::pow(theta, -static_cast<float>(i) / static_cast<float>(even_dim));
            const float angle = static_cast<float>(pos) * freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            table->cos[base + i] = c;
            table->cos[base + i + 1U] = c;
            table->sin[base + i] = s;
            table->sin[base + i + 1U] = s;
        }
    }

    return true;
}

void rope_table_free(RopeTable* table) {
    if (!table) {
        return;
    }
    aligned_free(table->cos);
    aligned_free(table->sin);
    table->cos = nullptr;
    table->sin = nullptr;
    table->max_position = 0;
    table->dim = 0;
}

void rope_table_apply(const RopeTable* table, float* x, std::uint32_t position) {
    if (!table || !x || !table->cos || !table->sin || position >= table->max_position) {
        return;
    }

    const std::uint32_t even_dim = table->dim;
    const std::size_t base = static_cast<std::size_t>(position) * even_dim;
    for (std::uint32_t i = 0; i < even_dim; i += 2U) {
        const float c = table->cos[base + i];
        const float s = table->sin[base + i];
        const float a = x[i];
        const float b = x[i + 1U];
        x[i] = (a * c) - (b * s);
        x[i + 1U] = (a * s) + (b * c);
    }
}

void softmax_inplace(float* x, std::uint32_t n) {
    if (!x || n == 0) {
        return;
    }

    float max_v = x[0];
    for (std::uint32_t i = 1; i < n; ++i) {
        max_v = std::max(max_v, x[i]);
    }

    float sum = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_v);
        sum += x[i];
    }

    const float inv_sum = 1.0f / sum;
    for (std::uint32_t i = 0; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

} // namespace ciot
