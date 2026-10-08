#include "../../include/Ciot.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ciot {

bool bpe_tokenizer_load(BpeTokenizer* tok, const char* merges_path, const char* vocab_path) {
    if (!tok || !merges_path || !vocab_path) return false;
    bpe_tokenizer_free(tok);

    // Load vocab: each line is "id<TAB>token"
    std::ifstream vf(vocab_path);
    if (!vf) return false;

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(vf, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    if (lines.size() < 256) return false;

    tok->vocab_size = static_cast<std::uint32_t>(lines.size());
    tok->vocab = static_cast<char**>(std::calloc(tok->vocab_size, sizeof(char*)));
    if (!tok->vocab) return false;

    for (std::uint32_t i = 0; i < lines.size(); ++i) {
        const std::string& ln = lines[i];
        std::size_t tab_pos = ln.find('\t');
        if (tab_pos == std::string::npos) { bpe_tokenizer_free(tok); return false; }
        std::string id_str = ln.substr(0, tab_pos);
        std::string token = ln.substr(tab_pos + 1);
        int id = std::atoi(id_str.c_str());
        // The id_str should match the line index, but we trust the file
        if (id < 0 || static_cast<std::uint32_t>(id) >= tok->vocab_size) continue;
        tok->vocab[id] = static_cast<char*>(std::malloc(token.size() + 1));
        if (!tok->vocab[id]) { bpe_tokenizer_free(tok); return false; }
        std::memcpy(tok->vocab[id], token.c_str(), token.size() + 1);
    }

    // Build byte-to-token lookup for raw bytes (GPT-2 style: byte b -> unicode char 256+b)
    // Most BPE vocabularies map byte values to unicode chars starting at 256
    std::int32_t byte_token[256];
    std::memset(byte_token, -1, sizeof(byte_token));
    for (std::uint32_t i = 0; i < tok->vocab_size; ++i) {
        const char* t = tok->vocab[i];
        int len = static_cast<int>(std::strlen(t));
        if (len >= 2 && static_cast<unsigned char>(t[0]) == 0xC4) {
            int b = static_cast<unsigned char>(t[1]) - 0x80 + 64;  // 2-byte UTF-8 for 0x80-0xBF = bytes 0-63
            if (b >= 0 && b < 64) byte_token[b] = static_cast<std::int32_t>(i);
        } else if (len >= 2 && static_cast<unsigned char>(t[0]) == 0xC5) {
            int b = static_cast<unsigned char>(t[1]) - 0x80 + 128; // 2-byte UTF-8 for 0xC0-0xFF = bytes 64-127
            if (b >= 64 && b < 128) byte_token[b] = static_cast<std::int32_t>(i);
        } else if (len >= 2 && static_cast<unsigned char>(t[0]) == 0xC3) {
            int b = static_cast<unsigned char>(t[1]) - 0x80;       // bytes 128-191
            if (b >= 128 && b < 192) byte_token[b] = static_cast<std::int32_t>(i);
        } else if (len == 1) {
            unsigned char c = static_cast<unsigned char>(t[0]);
            if (c < 128) byte_token[c] = static_cast<std::int32_t>(i);
        }
    }

    // Load merges: each line is "token_a token_b"
    std::ifstream mf(merges_path);
    if (!mf) { bpe_tokenizer_free(tok); return false; }

    // Allocate merge table: 256x256 grid of int ranks
    tok->merge_table = static_cast<int**>(std::calloc(256, sizeof(int*)));
    if (!tok->merge_table) { bpe_tokenizer_free(tok); return false; }
    for (int i = 0; i < 256; ++i) {
        tok->merge_table[i] = static_cast<int*>(std::calloc(256, sizeof(int)));
        if (!tok->merge_table[i]) { bpe_tokenizer_free(tok); return false; }
        for (int j = 0; j < 256; ++j) tok->merge_table[i][j] = -1;
    }

    std::string a, b;
    std::uint32_t rank = 0;
    while (mf >> a >> b) {
        if (rank >= 65536) break;
        int ai = -1, bi = -1;
        for (std::uint32_t i = 0; i < tok->vocab_size && (ai < 0 || bi < 0); ++i) {
            if (std::strcmp(tok->vocab[i], a.c_str()) == 0) ai = static_cast<int>(i);
            if (std::strcmp(tok->vocab[i], b.c_str()) == 0) bi = static_cast<int>(i);
        }
        if (ai >= 0 && ai < 256 && bi >= 0 && bi < 256 && rank < static_cast<std::uint32_t>(256 * 256))
            tok->merge_table[ai][bi] = static_cast<int>(rank);
        tok->num_merges++;
        rank++;
    }

    return true;
}

void bpe_tokenizer_free(BpeTokenizer* tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (std::uint32_t i = 0; i < tok->vocab_size; ++i) std::free(tok->vocab[i]);
        std::free(tok->vocab);
    }
    if (tok->merge_table) {
        for (int i = 0; i < 256; ++i) std::free(tok->merge_table[i]);
        std::free(tok->merge_table);
    }
    tok->vocab = nullptr;
    tok->merge_table = nullptr;
    tok->vocab_size = 0;
    tok->num_merges = 0;
}

std::uint32_t bpe_encode(const BpeTokenizer* tok, const char* text,
                          std::uint32_t* ids_out, std::uint32_t max_ids) {
    if (!tok || !text || !ids_out || max_ids == 0 || !tok->merge_table) return 0;

    const std::uint32_t max_symbols = max_ids * 4;
    std::uint32_t* symbols = static_cast<std::uint32_t*>(std::malloc(max_symbols * sizeof(std::uint32_t)));
    if (!symbols) return 0;
    std::uint32_t n = 0;

    // Convert text to initial byte tokens
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text);
    int len = static_cast<int>(std::strlen(text));
    for (int i = 0; i < len && n < max_symbols; ++i) {
        unsigned char b = bytes[i];
        // UTF-8 continuation bytes don't get their own token in GPT-2 style encoding
        // For simplicity, we map each byte to one of 256 placeholder IDs
        symbols[n++] = static_cast<std::uint32_t>(b);
    }
    if (n == 0) { std::free(symbols); return 0; }

    // BPE merge loop
    while (n > 1) {
        int best_rank = 1000000000;
        std::uint32_t best_pos = 0;
        for (std::uint32_t i = 0; i < n - 1; ++i) {
            std::uint32_t a = symbols[i], b = symbols[i + 1];
            if (a < 256 && b < 256) {
                int r = tok->merge_table[a][b];
                if (r >= 0 && r < best_rank) {
                    best_rank = r;
                    best_pos = i;
                }
            }
        }
        if (best_rank == 1000000000) break; // no more merges

        // Merge symbols[best_pos] and symbols[best_pos+1]
        std::uint32_t merged = static_cast<std::uint32_t>(256 + best_rank);
        if (merged >= tok->vocab_size) break;
        symbols[best_pos] = merged;
        for (std::uint32_t i = best_pos + 1; i < n - 1; ++i) symbols[i] = symbols[i + 1];
        n--;
    }

    // Output final IDs
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < n && count < max_ids; ++i)
        ids_out[count++] = symbols[i];

    std::free(symbols);
    return count;
}

const char* bpe_decode(const BpeTokenizer* tok, std::uint32_t id) {
    if (!tok || id >= tok->vocab_size) return "";
    return tok->vocab[id];
}

void bpe_decode_ids(const BpeTokenizer* tok, const std::uint32_t* ids,
                     std::uint32_t count, char* out, std::uint32_t max_out) {
    if (!tok || !ids || !out || max_out == 0) return;
    std::uint32_t pos = 0;
    for (std::uint32_t i = 0; i < count && pos < max_out; ++i) {
        const char* token = bpe_decode(tok, ids[i]);
        int len = static_cast<int>(std::strlen(token));
        for (int j = 0; j < len && pos < max_out - 1; ++j)
            out[pos++] = token[j];
    }
    out[pos] = '\0';
}

} // namespace ciot
