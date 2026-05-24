#include "../../include/Ciot.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ciot {

bool tokenizer_load(Tokenizer* tok, const char* vocab_path) {
    if (!tok || !vocab_path) return false;
    tokenizer_free(tok);

    std::ifstream file(vocab_path);
    if (!file) return false;

    std::vector<std::string> words;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            words.push_back(line);
        }
    }

    if (words.empty()) return false;

    tok->vocab_size = static_cast<std::uint32_t>(words.size());
    tok->tokens = static_cast<char**>(std::malloc(static_cast<std::size_t>(tok->vocab_size) * sizeof(char*)));
    if (!tok->tokens) { tok->vocab_size = 0; return false; }

    for (std::uint32_t i = 0; i < tok->vocab_size; ++i) {
        const std::size_t len = words[i].size() + 1;
        tok->tokens[i] = static_cast<char*>(std::malloc(len));
        if (!tok->tokens[i]) { tokenizer_free(tok); return false; }
        std::memcpy(tok->tokens[i], words[i].c_str(), len);
    }

    return true;
}

void tokenizer_free(Tokenizer* tok) {
    if (!tok) return;
    if (tok->tokens) {
        for (std::uint32_t i = 0; i < tok->vocab_size; ++i) {
            std::free(tok->tokens[i]);
        }
        std::free(tok->tokens);
        tok->tokens = nullptr;
    }
    tok->vocab_size = 0;
}

std::uint32_t tokenizer_encode(const Tokenizer* tok, const char* word) {
    if (!tok || !word) return 0;
    for (std::uint32_t i = 0; i < tok->vocab_size; ++i) {
        if (std::strcmp(tok->tokens[i], word) == 0) return i;
    }
    return 0;
}

const char* tokenizer_decode(const Tokenizer* tok, std::uint32_t id) {
    if (!tok || id >= tok->vocab_size) return "<unk>";
    return tok->tokens[id];
}

std::uint32_t tokenizer_encode_text(const Tokenizer* tok, const char* text,
                                    std::uint32_t* ids_out, std::uint32_t max_ids) {
    if (!tok || !text || !ids_out || max_ids == 0) return 0;

    std::string s(text);
    std::istringstream stream(s);
    std::string word;
    std::uint32_t count = 0;

    while (stream >> word && count < max_ids) {
        ids_out[count] = tokenizer_encode(tok, word.c_str());
        count++;
    }

    return count;
}

} // namespace ciot
