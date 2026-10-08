CXX ?= g++
# Default builds are portable. On an x86 CPU, -march=native enables AVX2 when the host supports it.
# Force AVX2 explicitly with: make avx2
# Force AVX-512 explicitly with: make avx512
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic -Iinclude
SIMD_CXXFLAGS ?= -std=c++17 -O3 -mavx2
AVX2_CXXFLAGS ?= -std=c++17 -O3 -mavx2 -Wall -Wextra -Wpedantic -Iinclude
AVX512_CXXFLAGS ?= -std=c++17 -O3 -mavx512f -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS ?=

CORE_SRC := src/kernels/ternary_simd.cpp src/linalg/linear.cpp src/model/ops.cpp src/model/tiny_transformer.cpp src/model/kv_cache.cpp src/model/mha_cache.cpp src/model/tokenizer.cpp src/model/bpe_tokenizer.cpp src/model/model_loader.cpp
APP_SRC := src/main.cpp $(CORE_SRC)
TEST_LINEAR_SRC := tests/test_linear.cpp $(CORE_SRC)
TEST_PROD_SRC := tests/test_production.cpp $(CORE_SRC)

BIN := bin/ciot
TEST_LINEAR_BIN := bin/test_linear
TEST_PROD_BIN := bin/test_production
TEST_SIMD_BIN := bin/test_simd

.PHONY: all clean test smoke bench simd-test avx2 avx512 avx2-hello dirs

all: $(BIN)

avx2: $(APP_SRC) include/Ciot.h | dirs
	$(CXX) $(AVX2_CXXFLAGS) $(APP_SRC) $(LDFLAGS) -o $(BIN)

avx512: $(APP_SRC) include/Ciot.h | dirs
	$(CXX) $(AVX512_CXXFLAGS) $(APP_SRC) $(LDFLAGS) -o $(BIN)

dirs:
	mkdir -p bin data

$(BIN): $(APP_SRC) include/Ciot.h | dirs
	$(CXX) $(CXXFLAGS) $(APP_SRC) $(LDFLAGS) -o $(BIN)

$(TEST_LINEAR_BIN): $(TEST_LINEAR_SRC) include/Ciot.h | dirs
	$(CXX) $(CXXFLAGS) $(TEST_LINEAR_SRC) $(LDFLAGS) -o $(TEST_LINEAR_BIN)

$(TEST_PROD_BIN): $(TEST_PROD_SRC) include/Ciot.h | dirs
	$(CXX) $(CXXFLAGS) $(TEST_PROD_SRC) $(LDFLAGS) -o $(TEST_PROD_BIN)

$(TEST_SIMD_BIN): tests/test_simd.cpp | dirs
	$(CXX) $(SIMD_CXXFLAGS) tests/test_simd.cpp -o $(TEST_SIMD_BIN)

simd-test: $(BIN)
	./$(BIN) --simd-test

avx2-hello: $(TEST_SIMD_BIN)
	./$(TEST_SIMD_BIN)

smoke: simd-test

test: $(TEST_LINEAR_BIN) $(TEST_PROD_BIN)
	./$(TEST_LINEAR_BIN)
	./$(TEST_PROD_BIN)

bench: $(BIN)
	./$(BIN) --bench-linear-pro 1024 1024 200 9 20

clean:
	rm -f $(BIN) $(TEST_LINEAR_BIN) $(TEST_SIMD_BIN)
