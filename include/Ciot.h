#ifndef CIOT_H
#define CIOT_H

#include <cstddef>
#include <cstdint>

namespace ciot {

constexpr std::size_t CIOT_CACHELINE = 64;
constexpr std::uint32_t CIOT_TERNARY_BLOCK = 64;

// ---------------------------------------------------------------
// Packed ternary matrix for W[x].
// Encoding is bit-plane based for SIMD friendliness:
//   pos_bits[row][block] bit k == 1 => weight +1
//   neg_bits[row][block] bit k == 1 => weight -1
//   both zero => weight 0
// `row_scale[row]` rescales the {-1,0,+1} row back toward the original floats.
// ---------------------------------------------------------------
struct TernaryMatrix {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t blocks64 = 0;
    std::uint64_t* pos_bits = nullptr;
    std::uint64_t* neg_bits = nullptr;
    float* row_scale = nullptr;
};

struct RopeTable {
    std::uint32_t max_position = 0;
    std::uint32_t dim = 0;
    float* cos = nullptr;
    float* sin = nullptr;
};

struct KVCache {
    std::uint32_t max_tokens = 0;
    std::uint32_t dim = 0;
    std::uint32_t used = 0;
    float* key = nullptr;
    float* value = nullptr;
};

// ---------------------------------------------------------------
// Multi-head KV cache: num_heads caches of head_dim each
// stored interleaved [head0_k | head0_v | head1_k | head1_v | ...]
// ---------------------------------------------------------------
struct MhaKVCache {
    std::uint32_t max_tokens = 0;
    std::uint32_t num_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t used = 0;
    float* buffer = nullptr;  // 2 * num_heads * max_tokens * head_dim
};

// ---------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------
struct CiotModelConfig {
    std::uint32_t dim = 0;
    std::uint32_t num_layers = 0;
    std::uint32_t num_heads = 0;
    std::uint32_t vocab_size = 0;
    std::uint32_t max_context = 0;
    float rope_theta = 10000.0f;
    float norm_eps = 1.0e-5f;
};

// ---------------------------------------------------------------
// BPE tokenizer (GPT-2 style byte-pair encoding)
// Loads merges.txt (one merge pair per line) and vocab.json (token -> id)
// ---------------------------------------------------------------
struct BpeTokenizer {
    std::uint32_t vocab_size = 0;
    char** vocab = nullptr;           // id -> token string
    std::uint32_t num_merges = 0;
    int** merge_table = nullptr;      // 256x256 -> rank lookup for encoding
};

bool bpe_tokenizer_load(BpeTokenizer* tok, const char* merges_path, const char* vocab_path);
void bpe_tokenizer_free(BpeTokenizer* tok);
std::uint32_t bpe_encode(const BpeTokenizer* tok, const char* text,
                          std::uint32_t* ids_out, std::uint32_t max_ids);
const char* bpe_decode(const BpeTokenizer* tok, std::uint32_t id);
void bpe_decode_ids(const BpeTokenizer* tok, const std::uint32_t* ids,
                     std::uint32_t count, char* out, std::uint32_t max_out);

// ---------------------------------------------------------------
// Simple word-level tokenizer (legacy)
// ---------------------------------------------------------------
struct Tokenizer {
    std::uint32_t vocab_size = 0;
    char** tokens = nullptr;
};

// ---------------------------------------------------------------
// Core allocator
// ---------------------------------------------------------------
void* aligned_malloc(std::size_t bytes, std::size_t alignment = CIOT_CACHELINE);
void aligned_free(void* ptr);

// ---------------------------------------------------------------
// Matrix ops
// ---------------------------------------------------------------
bool matrix_init(TernaryMatrix* matrix, std::uint32_t rows, std::uint32_t cols);
void matrix_free(TernaryMatrix* matrix);
void matrix_zero(TernaryMatrix* matrix);
bool matrix_load_bits(TernaryMatrix* matrix, const char* path);
void matrix_set_ternary(TernaryMatrix* matrix, std::uint32_t row, std::uint32_t col, std::int8_t value);
std::int8_t matrix_get_ternary(const TernaryMatrix* matrix, std::uint32_t row, std::uint32_t col);

void matvec_ternary_ref(const TernaryMatrix* matrix, const float* x, float* y);
void matvec_ternary_simd(const TernaryMatrix* matrix, const float* x, float* y);
void matmat_ternary_simd(const TernaryMatrix* matrix, const float* x, std::uint32_t batch, float* y);
const char* simd_backend_name();

// ---------------------------------------------------------------
// Transformer primitives
// ---------------------------------------------------------------
void rmsnorm(float* out, const float* x, const float* weight, std::uint32_t n, float eps);
void rope_pairwise(float* x, std::uint32_t dim, std::uint32_t position, float theta);
void softmax_inplace(float* x, std::uint32_t n);
bool rope_table_init(RopeTable* table, std::uint32_t max_position, std::uint32_t dim, float theta);
void rope_table_free(RopeTable* table);
void rope_table_apply(const RopeTable* table, float* x, std::uint32_t position);

// ---------------------------------------------------------------
// Single-head KV cache
// ---------------------------------------------------------------
bool kv_cache_init(KVCache* cache, std::uint32_t max_tokens, std::uint32_t dim);
void kv_cache_free(KVCache* cache);
void kv_cache_reset(KVCache* cache);
void kv_cache_append(KVCache* cache, const float* key, const float* value);
void attention_decode_1head(const KVCache* cache, const float* query, float* out,
                            float* score_scratch, std::uint32_t dim);

// ---------------------------------------------------------------
// Multi-head KV cache + attention
// ---------------------------------------------------------------
bool mha_kv_cache_init(MhaKVCache* cache, std::uint32_t max_tokens,
                       std::uint32_t num_heads, std::uint32_t head_dim);
void mha_kv_cache_free(MhaKVCache* cache);
void mha_kv_cache_reset(MhaKVCache* cache);
void mha_kv_cache_append(MhaKVCache* cache, const float* keys, const float* values);

void mha_attention_decode(const MhaKVCache* cache, const float* query,
                          float* out, float* score_scratch,
                          std::uint32_t num_heads, std::uint32_t head_dim);

// ---------------------------------------------------------------
// Full decode-time transformer block with multi-head attention
// scratch must hold at least (7 + num_heads * 2) * dim + max_context * num_heads floats
// ---------------------------------------------------------------
void transformer_block_decode_mha(float* x,
                                  const TernaryMatrix* wq,
                                  const TernaryMatrix* wk,
                                  const TernaryMatrix* wv,
                                  const TernaryMatrix* wo,
                                  const TernaryMatrix* w1,
                                  const TernaryMatrix* w2,
                                  const float* norm1_weight,
                                  const float* norm2_weight,
                                  const RopeTable* rope,
                                  MhaKVCache* cache,
                                  float* scratch,
                                  std::uint32_t dim,
                                  std::uint32_t num_heads,
                                  std::uint32_t position,
                                  float eps);

// ---------------------------------------------------------------
// Legacy single-head decode block (for backward compat)
// ---------------------------------------------------------------
void transformer_block_1tok(float* x,
                            const TernaryMatrix* wq, const TernaryMatrix* wk,
                            const TernaryMatrix* wv, const TernaryMatrix* wo,
                            const TernaryMatrix* w1, const TernaryMatrix* w2,
                            const float* norm1_weight, const float* norm2_weight,
                            float* scratch, std::uint32_t dim, float eps);

void transformer_block_decode_1tok(float* x,
                                   const TernaryMatrix* wq, const TernaryMatrix* wk,
                                   const TernaryMatrix* wv, const TernaryMatrix* wo,
                                   const TernaryMatrix* w1, const TernaryMatrix* w2,
                                   const float* norm1_weight, const float* norm2_weight,
                                   const RopeTable* rope, KVCache* cache,
                                   float* scratch, std::uint32_t dim,
                                   std::uint32_t position, float eps);

// ---------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------
bool tokenizer_load(Tokenizer* tok, const char* vocab_path);
void tokenizer_free(Tokenizer* tok);
std::uint32_t tokenizer_encode(const Tokenizer* tok, const char* word);
const char* tokenizer_decode(const Tokenizer* tok, std::uint32_t id);
std::uint32_t tokenizer_encode_text(const Tokenizer* tok, const char* text,
                                    std::uint32_t* ids_out, std::uint32_t max_ids);

// ---------------------------------------------------------------
// Model loader: loads config + all weight matrices from a directory
// Model dir layout:
//   config.ciot  - JSON-ish key=value
//   embed.bits   - vocab_size x dim embedding matrix
//   layer0/wq.bits ... layerN/w2.bits, layerN/norm1.f32, layerN/norm2.f32
//   lm_head.bits - dim x vocab_size output projection
// ---------------------------------------------------------------
bool model_config_load(CiotModelConfig* cfg, const char* path);
bool model_load_weights(const CiotModelConfig* cfg, const char* dir,
                        TernaryMatrix* embed, TernaryMatrix* lm_head,
                        TernaryMatrix* wq, TernaryMatrix* wk,
                        TernaryMatrix* wv, TernaryMatrix* wo,
                        TernaryMatrix* w1, TernaryMatrix* w2,
                        float* norm1_weight, float* norm2_weight);

std::uint64_t ticks_ns();

} // namespace ciot

#endif // CIOT_H
