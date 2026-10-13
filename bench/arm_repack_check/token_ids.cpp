// Greedy IDs through the public API, without text or chat parsing.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: ciot-token-check MODEL PROMPT_FILE TOKEN_COUNT THREADS\n");
        return 1;
    }
    const int count = std::atoi(argv[3]);
    const int threads = std::atoi(argv[4]);
    if (count < 1 || count > 128 || threads < 1 || threads > 8) { return 1; }
    std::ifstream file(argv[2], std::ios::binary);
    if (!file) { std::fprintf(stderr, "cannot read prompt\n"); return 1; }
    const std::string prompt((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context * context = llama_init_from_model(model, cp);
    if (!context) { std::fprintf(stderr, "context load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int needed = -llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
    if (needed < 1 || needed > 128) { std::fprintf(stderr, "prompt must be 1..128 tokens\n"); return 1; }
    std::vector<llama_token> input(needed);
    if (llama_tokenize(vocab, prompt.data(), int(prompt.size()), input.data(), needed, true, true) != needed) { return 1; }
    if (llama_decode(context, llama_batch_get_one(input.data(), needed)) != 0) { return 1; }
    llama_sampler * greedy = llama_sampler_init_greedy();
    std::vector<llama_token> output;
    for (int i = 0; i < count; ++i) {
        llama_token token = llama_sampler_sample(greedy, context, -1);
        output.push_back(token);
        llama_sampler_accept(greedy, token);
        if (llama_vocab_is_eog(vocab, token) || i + 1 == count) { break; }
        if (llama_decode(context, llama_batch_get_one(&token, 1)) != 0) { return 1; }
    }
    std::printf("{\"tokens\":[");
    for (size_t i = 0; i < output.size(); ++i) { std::printf("%s%d", i ? "," : "", output[i]); }
    std::puts("]}");
    llama_sampler_free(greedy);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
