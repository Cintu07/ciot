#include "../../include/Ciot.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace ciot {

void* aligned_malloc(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        bytes = alignment;
    }

#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free(void* ptr) {
    if (!ptr) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

bool matrix_init(TernaryMatrix* matrix, std::uint32_t rows, std::uint32_t cols) {
    if (!matrix || rows == 0 || cols == 0) {
        return false;
    }

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->blocks64 = (cols + CIOT_TERNARY_BLOCK - 1U) / CIOT_TERNARY_BLOCK;

    const std::size_t words = static_cast<std::size_t>(rows) * matrix->blocks64;
    matrix->pos_bits = static_cast<std::uint64_t*>(aligned_malloc(words * sizeof(std::uint64_t)));
    matrix->neg_bits = static_cast<std::uint64_t*>(aligned_malloc(words * sizeof(std::uint64_t)));
    matrix->row_scale = static_cast<float*>(aligned_malloc(static_cast<std::size_t>(rows) * sizeof(float)));

    if (!matrix->pos_bits || !matrix->neg_bits || !matrix->row_scale) {
        matrix_free(matrix);
        return false;
    }

    std::memset(matrix->pos_bits, 0, words * sizeof(std::uint64_t));
    std::memset(matrix->neg_bits, 0, words * sizeof(std::uint64_t));
    for (std::uint32_t r = 0; r < rows; ++r) {
        matrix->row_scale[r] = 1.0f;
    }
    return true;
}

void matrix_free(TernaryMatrix* matrix) {
    if (!matrix) {
        return;
    }

    aligned_free(matrix->pos_bits);
    aligned_free(matrix->neg_bits);
    aligned_free(matrix->row_scale);
    matrix->rows = 0;
    matrix->cols = 0;
    matrix->blocks64 = 0;
    matrix->pos_bits = nullptr;
    matrix->neg_bits = nullptr;
    matrix->row_scale = nullptr;
}

void matrix_zero(TernaryMatrix* matrix) {
    if (!matrix || !matrix->pos_bits || !matrix->neg_bits) {
        return;
    }
    const std::size_t words = static_cast<std::size_t>(matrix->rows) * matrix->blocks64;
    std::memset(matrix->pos_bits, 0, words * sizeof(std::uint64_t));
    std::memset(matrix->neg_bits, 0, words * sizeof(std::uint64_t));
}

bool matrix_load_bits(TernaryMatrix* matrix, const char* path) {
    if (!matrix || !path) {
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[8] = {};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t blocks64 = 0;

    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in.read(reinterpret_cast<char*>(&blocks64), sizeof(blocks64));

    if (!in || std::memcmp(magic, "CIOTBIT1", 8) != 0 || rows == 0 || cols == 0) {
        return false;
    }

    const std::uint32_t expected_blocks = (cols + CIOT_TERNARY_BLOCK - 1U) / CIOT_TERNARY_BLOCK;
    if (blocks64 != expected_blocks) {
        return false;
    }

    TernaryMatrix loaded;
    if (!matrix_init(&loaded, rows, cols)) {
        return false;
    }

    const std::size_t words = static_cast<std::size_t>(rows) * blocks64;
    in.read(reinterpret_cast<char*>(loaded.row_scale), static_cast<std::streamsize>(static_cast<std::size_t>(rows) * sizeof(float)));
    in.read(reinterpret_cast<char*>(loaded.pos_bits), static_cast<std::streamsize>(words * sizeof(std::uint64_t)));
    in.read(reinterpret_cast<char*>(loaded.neg_bits), static_cast<std::streamsize>(words * sizeof(std::uint64_t)));

    if (!in) {
        matrix_free(&loaded);
        return false;
    }

    matrix_free(matrix);
    *matrix = loaded;
    return true;
}

void matrix_set_ternary(TernaryMatrix* matrix, std::uint32_t row, std::uint32_t col, std::int8_t value) {
    if (!matrix || row >= matrix->rows || col >= matrix->cols) {
        return;
    }

    const std::uint32_t block = col >> 6U;
    const std::uint32_t bit = col & 63U;
    const std::size_t index = static_cast<std::size_t>(row) * matrix->blocks64 + block;
    const std::uint64_t mask = 1ULL << bit;

    matrix->pos_bits[index] &= ~mask;
    matrix->neg_bits[index] &= ~mask;

    if (value > 0) {
        matrix->pos_bits[index] |= mask;
    } else if (value < 0) {
        matrix->neg_bits[index] |= mask;
    }
}

std::int8_t matrix_get_ternary(const TernaryMatrix* matrix, std::uint32_t row, std::uint32_t col) {
    if (!matrix || row >= matrix->rows || col >= matrix->cols) {
        return 0;
    }

    const std::uint32_t block = col >> 6U;
    const std::uint32_t bit = col & 63U;
    const std::size_t index = static_cast<std::size_t>(row) * matrix->blocks64 + block;
    const std::uint64_t mask = 1ULL << bit;
    const std::uint32_t pos = (matrix->pos_bits[index] & mask) != 0ULL;
    const std::uint32_t neg = (matrix->neg_bits[index] & mask) != 0ULL;
    return static_cast<std::int8_t>(static_cast<int>(pos) - static_cast<int>(neg));
}

void matvec_ternary_ref(const TernaryMatrix* matrix, const float* x, float* y) {
    if (!matrix || !x || !y) {
        return;
    }

    for (std::uint32_t r = 0; r < matrix->rows; ++r) {
        float sum = 0.0f;
        const std::size_t row_base = static_cast<std::size_t>(r) * matrix->blocks64;
        for (std::uint32_t c = 0; c < matrix->cols; ++c) {
            const std::uint32_t block = c >> 6U;
            const std::uint32_t bit = c & 63U;
            const std::uint64_t bit_mask = 1ULL << bit;
            const std::uint32_t pos = (matrix->pos_bits[row_base + block] & bit_mask) != 0ULL;
            const std::uint32_t neg = (matrix->neg_bits[row_base + block] & bit_mask) != 0ULL;
            const int sign = static_cast<int>(pos) - static_cast<int>(neg);
            sum += static_cast<float>(sign) * x[c];
        }
        y[r] = sum * matrix->row_scale[r];
    }
}

std::uint64_t ticks_ns() {
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

} // namespace ciot
