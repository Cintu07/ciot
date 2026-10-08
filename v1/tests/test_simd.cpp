#include <immintrin.h>
#include <iostream>

int main() {
    __m256i v1 = _mm256_set1_epi32(10);
    __m256i v2 = _mm256_set1_epi32(20);
    __m256i result = _mm256_add_epi32(v1, v2);

    int out[8];
    _mm256_storeu_si256((__m256i*)out, result);

    std::cout << "SIMD Test: " << out[0] << " (Should be 30)" << std::endl;
    return out[0] == 30 ? 0 : 1;
}
