#include "../../include/Ciot.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ciot {

bool model_config_load(CiotModelConfig* cfg, const char* path) {
    if (!cfg || !path) return false;

    FILE* f = std::fopen(path, "r");
    if (!f) return false;

    char key[128];
    int val = 0;
    while (std::fscanf(f, "%127s %d", key, &val) == 2) {
        if (std::strcmp(key, "dim") == 0) cfg->dim = static_cast<std::uint32_t>(val);
        else if (std::strcmp(key, "num_layers") == 0) cfg->num_layers = static_cast<std::uint32_t>(val);
        else if (std::strcmp(key, "num_heads") == 0) cfg->num_heads = static_cast<std::uint32_t>(val);
        else if (std::strcmp(key, "vocab_size") == 0) cfg->vocab_size = static_cast<std::uint32_t>(val);
        else if (std::strcmp(key, "max_context") == 0) cfg->max_context = static_cast<std::uint32_t>(val);
    }
    std::fclose(f);

    if (cfg->dim == 0 || cfg->num_layers == 0 || cfg->num_heads == 0 || cfg->vocab_size == 0) {
        return false;
    }
    if (cfg->max_context == 0) cfg->max_context = 64;
    return true;
}

bool model_load_weights(const CiotModelConfig* cfg, const char* dir,
                        TernaryMatrix* embed, TernaryMatrix* lm_head,
                        TernaryMatrix* wq, TernaryMatrix* wk,
                        TernaryMatrix* wv, TernaryMatrix* wo,
                        TernaryMatrix* w1, TernaryMatrix* w2,
                        float* norm1_weight, float* norm2_weight) {
    if (!cfg || !dir || !embed || !lm_head || !wq || !wk || !wv || !wo || !w1 || !w2) return false;
    if (!norm1_weight || !norm2_weight) return false;

    char path[512];

    // Embedding: vocab_size x dim
    std::snprintf(path, sizeof(path), "%s/embed.bits", dir);
    if (!matrix_load_bits(embed, path)) return false;

    // LM head: dim x vocab_size
    std::snprintf(path, sizeof(path), "%s/lm_head.bits", dir);
    if (!matrix_load_bits(lm_head, path)) return false;

    for (std::uint32_t l = 0; l < cfg->num_layers; ++l) {
        const char* names[6] = {"wq", "wk", "wv", "wo", "w1", "w2"};
        TernaryMatrix* mats[6] = {&wq[l], &wk[l], &wv[l], &wo[l], &w1[l], &w2[l]};
        for (int m = 0; m < 6; ++m) {
            std::snprintf(path, sizeof(path), "%s/layer%u/%s.bits", dir, l, names[m]);
            if (!matrix_load_bits(mats[m], path)) return false;
        }
        std::snprintf(path, sizeof(path), "%s/layer%u/norm1.f32", dir, l);
        FILE* f = std::fopen(path, "rb");
        if (f) {
            std::fread(norm1_weight + (static_cast<std::size_t>(l) * cfg->dim), sizeof(float), cfg->dim, f);
            std::fclose(f);
        } else {
            for (std::uint32_t i = 0; i < cfg->dim; ++i)
                norm1_weight[static_cast<std::size_t>(l) * cfg->dim + i] = 1.0f;
        }
        std::snprintf(path, sizeof(path), "%s/layer%u/norm2.f32", dir, l);
        f = std::fopen(path, "rb");
        if (f) {
            std::fread(norm2_weight + (static_cast<std::size_t>(l) * cfg->dim), sizeof(float), cfg->dim, f);
            std::fclose(f);
        } else {
            for (std::uint32_t i = 0; i < cfg->dim; ++i)
                norm2_weight[static_cast<std::size_t>(l) * cfg->dim + i] = 1.0f;
        }
    }
    return true;
}

} // namespace ciot
