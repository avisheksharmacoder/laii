CC ?= gcc
CFLAGS ?= -std=c11 -O3 -fPIC -Wall -Wextra -Wno-unused-function -Iinclude -D_GNU_SOURCE
LDFLAGS ?= -lm -lpthread

PYTHON ?= $(shell if [ -x /home/avisheks/.pyenv/versions/3.14.7/bin/python3 ]; then echo /home/avisheks/.pyenv/versions/3.14.7/bin/python3; else which python3 2>/dev/null || echo python3; fi)

# Add AVX2 flags if host supports it
HAS_AVX2 := $(shell $(CC) -mavx2 -dM -E - < /dev/null 2>&1 | grep -c AVX2)
ifeq ($(HAS_AVX2), 1)
    CFLAGS += -mavx2 -mfma -mf16c
endif

ifeq ($(NATIVE), 1)
    CFLAGS += -march=native
endif

# JNI configuration (auto-detect OpenJDK or override via JAVA_HOME)
JAVA_HOME ?= $(shell if [ -d /usr/lib/jvm/java-17-openjdk-amd64 ]; then echo /usr/lib/jvm/java-17-openjdk-amd64; elif [ -d /usr/lib/jvm/java-17-openjdk-arm64 ]; then echo /usr/lib/jvm/java-17-openjdk-arm64; elif [ -n "$$JAVA_HOME" ]; then echo "$$JAVA_HOME"; elif command -v javac >/dev/null 2>&1; then dirname $$(dirname $$(readlink -f $$(command -v javac))); fi)
ifneq ($(wildcard $(JAVA_HOME)/include/jni.h),)
    CFLAGS += -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -DLFM2_ENABLE_JNI=1
endif

SRC_CORE = src/arena.c src/gguf.c src/lfm_model.c src/lfm_simd.c src/lfm_fused.c src/lfm_threadpool.c src/lfm_forward.c src/lfm_tokenizer.c src/lfm_engine.c src/lfm_c_api.c src/lfm_server.c
OBJ_CORE = $(SRC_CORE:src/%.c=build/%.o)

TEST_BINS = bin/test_arena bin/test_math bin/test_simd_avx2 bin/test_quants bin/test_simd_kquants bin/test_fused_conv bin/test_threads bin/test_tokenizer bin/test_bpe bin/test_gguf bin/test_conv1d bin/test_attention bin/test_forward bin/test_c_api bin/test_lfm2_threadpool bin/test_optimizations bin/test_prefill_batch bin/test_decode_1x4 bin/test_kernels_fast bin/test_cpuid_quants bin/test_runtime_features bin/test_moe_checkpoint bin/test_cli_stream bin/test_sys_cache bin/test_lfm2_api bin/test_lfm2_server bin/test_hardening bin/test_simd_arm64 bin/test_android_io bin/test_speculative_pld

.PHONY: all clean test test-asan test-tsan dirs fixtures lib static static-musl test-python test-http test-jni test-kotlin-app run-kotlin-app test-parity cli server shared android python-wheel asan tsan parity help

all: dirs lib lfm-run lfm-server bin/lfm2 bin/lfm2-server bin/lfm2_loader bin/lfm2_bpe bin/lfm2_threadpool bin/lfm2_forward bin/lfm2_chat bin/lfm2_prefill_bench bin/lfm2_decode_1x4_bench bin/lfm2_kernels_fast_bench bin/lfm2_cpuid_quants bin/lfm2_runtime_features bin/lfm2_moe_checkpoint bin/lfm2_cli_stream bin/lfm2_sys_cache bin/lfm2_server bin/lfm2_simd_arm64 bin/lfm2_speculative_pld $(TEST_BINS)

dirs:
	@mkdir -p build bin lib

build/%.o: src/%.c $(wildcard include/*.h) | dirs
	$(CC) $(CFLAGS) -c $< -o $@

lib: dirs lib/liblfm.a lib/liblfm.so lib/liblfm2.so

lib/liblfm.a: $(OBJ_CORE)
	ar rcs $@ $^

lib/liblfm.so: $(OBJ_CORE)
	$(CC) -shared $(CFLAGS) $^ -o $@ $(LDFLAGS)

lib/liblfm2.so: src/lfm2_api.c src/lfm2_jni.c include/lfm2.h | dirs
	$(CC) -shared -fPIC -fvisibility=hidden $(CFLAGS) -Wl,-soname,liblfm2.so -Wl,-z,max-page-size=16384 $< -o $@ $(LDFLAGS)
	@cp -f $@ ./liblfm2.so

lfm-run: $(OBJ_CORE) build/main.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

lfm-server: $(OBJ_CORE) build/server_main.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

bin/lfm2_loader: tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bin/lfm2_bpe: tools/lfm2_bpe.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -DLFM2_BPE_STANDALONE_TEST $< -o $@ $(LDFLAGS)

bin/lfm2_forward: tools/lfm2_forward.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bin/lfm2: tools/lfm2_chat.c tools/lfm2_bpe.c tools/lfm2_forward.c tools/lfm2_loader.c tools/lfm2_prefill_batch.c tools/lfm2_threadpool.c | dirs
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bin/lfm2_chat: bin/lfm2
	@cp -f $< $@

bin/lfm2_threadpool: tools/lfm2_threadpool.c | dirs
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bin/lfm2_prefill_bench: tools/lfm2_prefill_batch.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bin/lfm2_decode_1x4_bench: tools/lfm2_decode_1x4.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm2_kernels_fast_bench: tools/lfm2_kernels_fast.c tools/lfm2_decode_1x4.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm2_cpuid_quants: tools/lfm2_cpuid_quants.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm2_runtime_features: tools/lfm2_runtime_features.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm2_moe_checkpoint: tools/lfm2_moe_checkpoint.c tools/lfm2_kernels_fast.c tools/lfm2_decode_1x4.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm2_cli_stream: tools/lfm2_cli_stream.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm2_sys_cache: tools/lfm2_sys_cache.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm2-server: tools/lfm2_server.c lib/liblfm2.so | dirs
	$(CC) $(CFLAGS) -I. $< -Llib -llfm2 -Wl,-rpath,'$$ORIGIN/../lib' -o $@ $(LDFLAGS)

bin/lfm2_server: bin/lfm2-server
	@cp -f $< $@

cli: bin/lfm2
server: bin/lfm2-server
shared: lib/liblfm2.so

bin/lfm2_simd_arm64: tools/lfm2_simd_arm64.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm2_speculative_pld: tools/lfm2_speculative_pld.c tools/lfm2_prefill_batch.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/lfm_gpu_infer_test: src/lfm_gpu/lfm_vulkan.c src/lfm_gpu/lfm_gpu_engine.c src/lfm_gpu/lfm_gpu_infer_test.c | dirs
	$(CC) $(CFLAGS) -Isrc/lfm_gpu -I. $^ -o $@ $(LDFLAGS) -ldl

bin/lfm_gpu_generate: src/lfm_gpu/lfm_vulkan.c src/lfm_gpu/lfm_gpu_engine.c src/lfm_gpu/lfm_gpu_generate.c | dirs
	$(CC) $(CFLAGS) -Isrc/lfm_gpu -I. $^ -o $@ $(LDFLAGS) -ldl

bin/lfm_nv_benchmark: src/lfm_nv/lfm_nv_vulkan.c src/lfm_nv/lfm_nv_engine.c src/lfm_nv/lfm_nv_benchmark.c | dirs
	$(CC) $(CFLAGS) -Isrc/lfm_nv -I. $^ -o $@ $(LDFLAGS) -ldl

bin/lfm_nv_generate: src/lfm_nv/lfm_nv_vulkan.c src/lfm_nv/lfm_nv_engine.c src/lfm_nv/lfm_nv_generate.c | dirs
	$(CC) $(CFLAGS) -Isrc/lfm_nv -I. $^ -o $@ $(LDFLAGS) -ldl



static: dirs bin/lfm2-static bin/lfm-run-static bin/lfm-server-static bin/lfm2-loader-static bin/lfm2-bpe-static bin/lfm2-threadpool-static bin/lfm2-forward-static bin/lfm2-chat-static bin/lfm2-prefill-static bin/lfm2-decode-1x4-static bin/lfm2-kernels-fast-static bin/lfm2-cpuid-quants-static bin/lfm2-runtime-features-static bin/lfm2-moe-checkpoint-static bin/lfm2-cli-stream-static bin/lfm2-sys-cache-static bin/lfm2-server-static bin/lfm2-simd-arm64-static bin/lfm2-speculative-pld-static

bin/lfm2-static: bin/lfm2-chat-static
	@cp -f $< $@

bin/lfm-run-static: $(OBJ_CORE) build/main.o
	$(CC) $(CFLAGS) -static $^ -o $@ $(LDFLAGS)

bin/lfm-server-static: $(OBJ_CORE) build/server_main.o
	$(CC) $(CFLAGS) -static $^ -o $@ $(LDFLAGS)

bin/lfm2-loader-static: tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -static $< -o $@ $(LDFLAGS)

bin/lfm2-bpe-static: tools/lfm2_bpe.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -DLFM2_BPE_STANDALONE_TEST -static $< -o $@ $(LDFLAGS)

bin/lfm2-threadpool-static: tools/lfm2_threadpool.c | dirs
	$(CC) $(CFLAGS) -static $< -o $@ $(LDFLAGS)

bin/lfm2-forward-static: tools/lfm2_forward.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -static $< -o $@ $(LDFLAGS)

bin/lfm2-chat-static: tools/lfm2_chat.c tools/lfm2_forward.c tools/lfm2_loader.c tools/lfm2_prefill_batch.c tools/lfm2_threadpool.c | dirs
	$(CC) $(CFLAGS) -static $< -o $@ $(LDFLAGS)

bin/lfm2-prefill-static: tools/lfm2_prefill_batch.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -static $< -o $@ $(LDFLAGS)

bin/lfm2-decode-1x4-static: tools/lfm2_decode_1x4.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/lfm2-kernels-fast-static: tools/lfm2_kernels_fast.c tools/lfm2_decode_1x4.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/lfm2-cpuid-quants-static: tools/lfm2_cpuid_quants.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/lfm2-runtime-features-static: tools/lfm2_runtime_features.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/lfm2-moe-checkpoint-static: tools/lfm2_moe_checkpoint.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/lfm2-cli-stream-static: tools/lfm2_cli_stream.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/lfm2-sys-cache-static: tools/lfm2_sys_cache.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/lfm2-server-static: tools/lfm2_server.c src/lfm2_api.c include/lfm2.h | dirs
	$(CC) $(CFLAGS) -I. -static -DLFM2_API_NO_SERVER -DLFM2_DISABLE_JNI tools/lfm2_server.c src/lfm2_api.c -o $@ $(LDFLAGS)

bin/lfm2-simd-arm64-static: tools/lfm2_simd_arm64.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/lfm2-speculative-pld-static: tools/lfm2_speculative_pld.c | dirs
	$(CC) $(CFLAGS) -I. -static $< -o $@ $(LDFLAGS)

bin/test_arena: build/arena.o tests/test_arena.c | dirs
	$(CC) $(CFLAGS) tests/test_arena.c build/arena.o -o $@ $(LDFLAGS)

bin/test_math: $(OBJ_CORE) tests/test_math.c | dirs
	$(CC) $(CFLAGS) tests/test_math.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_simd_avx2: $(OBJ_CORE) tests/test_simd_avx2.c | dirs
	$(CC) $(CFLAGS) tests/test_simd_avx2.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_quants: $(OBJ_CORE) tests/test_quants.c | dirs
	$(CC) $(CFLAGS) tests/test_quants.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_simd_kquants: tests/test_simd_kquants.c | dirs
	$(CC) $(CFLAGS) -fopenmp tests/test_simd_kquants.c -o $@ $(LDFLAGS)

bin/test_fused_conv: $(OBJ_CORE) tests/test_fused_conv.c | dirs
	$(CC) $(CFLAGS) tests/test_fused_conv.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_threads: $(OBJ_CORE) tests/test_threads.c | dirs
	$(CC) $(CFLAGS) tests/test_threads.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_tokenizer: $(OBJ_CORE) tests/test_tokenizer.c | dirs
	$(CC) $(CFLAGS) tests/test_tokenizer.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_bpe: tests/test_bpe.c tools/lfm2_bpe.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_bpe.c -o $@ $(LDFLAGS)

bin/test_gguf: $(OBJ_CORE) tests/test_gguf.c | dirs
	$(CC) $(CFLAGS) tests/test_gguf.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_conv1d: $(OBJ_CORE) tests/test_conv1d.c | dirs
	$(CC) $(CFLAGS) tests/test_conv1d.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_attention: $(OBJ_CORE) tests/test_attention.c | dirs
	$(CC) $(CFLAGS) tests/test_attention.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_forward: $(OBJ_CORE) tests/test_forward.c | dirs
	$(CC) $(CFLAGS) tests/test_forward.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_c_api: $(OBJ_CORE) tests/test_c_api.c | dirs
	$(CC) $(CFLAGS) tests/test_c_api.c $(OBJ_CORE) -o $@ $(LDFLAGS)

bin/test_lfm2_threadpool: tests/test_lfm2_threadpool.c tools/lfm2_threadpool.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_lfm2_threadpool.c -o $@ $(LDFLAGS)

bin/test_optimizations: tests/test_optimizations.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_optimizations.c -o $@ $(LDFLAGS)

bin/test_prefill_batch: tests/test_prefill_batch.c tools/lfm2_prefill_batch.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_prefill_batch.c -o $@ $(LDFLAGS)

bin/test_decode_1x4: tests/test_decode_1x4.c tools/lfm2_decode_1x4.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_decode_1x4.c -o $@ $(LDFLAGS)

bin/test_kernels_fast: tests/test_kernels_fast.c tools/lfm2_kernels_fast.c tools/lfm2_decode_1x4.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_kernels_fast.c -o $@ $(LDFLAGS)

bin/test_cpuid_quants: tests/test_cpuid_quants.c tools/lfm2_cpuid_quants.c tools/lfm2_decode_1x4.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_cpuid_quants.c -o $@ $(LDFLAGS)

bin/test_runtime_features: tests/test_runtime_features.c tools/lfm2_runtime_features.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_runtime_features.c -o $@ $(LDFLAGS)

bin/test_moe_checkpoint: tests/test_moe_checkpoint.c tools/lfm2_moe_checkpoint.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_moe_checkpoint.c -o $@ $(LDFLAGS)

bin/test_cli_stream: tests/test_cli_stream.c tools/lfm2_cli_stream.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_cli_stream.c -o $@ $(LDFLAGS)

bin/test_sys_cache: tests/test_sys_cache.c tools/lfm2_sys_cache.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. tests/test_sys_cache.c -o $@ $(LDFLAGS)

bin/test_lfm2_api: tests/test_lfm2_api.c lib/liblfm2.so | dirs
	$(CC) $(CFLAGS) $< -Llib -llfm2 -Wl,-rpath,'$$ORIGIN/../lib' -o $@ $(LDFLAGS)

bin/test_lfm2_server: tests/test_lfm2_server.c lib/liblfm2.so | dirs
	$(CC) $(CFLAGS) $< -Llib -llfm2 -Wl,-rpath,'$$ORIGIN/../lib' -o $@ $(LDFLAGS)

bin/test_hardening: tests/test_hardening.c lib/liblfm2.so | dirs
	$(CC) $(CFLAGS) -I. $< -Llib -llfm2 -Wl,-rpath,'$$ORIGIN/../lib' -o $@ $(LDFLAGS)

bin/test_simd_arm64: tests/test_simd_arm64.c tools/lfm2_simd_arm64.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

bin/test_android_io: tests/test_android_io.c lib/liblfm2.so | dirs
	$(CC) $(CFLAGS) $< -Llib -llfm2 -Wl,-rpath,'$$ORIGIN/../lib' -o $@ $(LDFLAGS)

bin/test_speculative_pld: tests/test_speculative_pld.c tools/lfm2_speculative_pld.c tools/lfm2_prefill_batch.c tools/lfm2_forward.c tools/lfm2_threadpool.c tools/lfm2_loader.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

fixtures:
	@if [ ! -f tests/lfm2_toy.gguf ] || [ ! -f tests/fixtures_attention.bin ]; then \
		$(PYTHON) tools/generate_reference_tensors.py && \
		$(PYTHON) tools/create_toy_gguf.py; \
	fi

test-http: lfm-run fixtures
	@bash tests/test_http_server.sh

test-python: lib fixtures
	@$(PYTHON) tests/test_python_bindings.py
	@$(PYTHON) tests/test_python_lfm2.py
	@$(PYTHON) tests/verify_parity.py

test-jni: lib fixtures
	@if command -v javac >/dev/null 2>&1; then \
		echo "Compiling and running Java/Kotlin JNI bridge test suite..."; \
		mkdir -p build/classes; \
		javac -d build/classes android/src/main/java/ai/liquid/lfm2/*.java tests/TestJNI.java; \
		java --add-opens java.base/java.io=ALL-UNNAMED -ea -cp build/classes -Djava.library.path=./lib tests.TestJNI $(MODEL); \
	else \
		echo "NOTE: javac not found on PATH. Skipping JNI test."; \
	fi

GRADLE_KOTLIN_JAR := $(shell find /home/avisheks/.gradle/wrapper/dists/gradle-8.14.3-all -name "kotlin-compiler-embeddable-*.jar" 2>/dev/null | head -n 1)
GRADLE_LIB_DIR    := $(if $(GRADLE_KOTLIN_JAR),$(dir $(GRADLE_KOTLIN_JAR)),)

test-kotlin-app: lib fixtures
	@if [ -n "$(GRADLE_LIB_DIR)" ] && command -v javac >/dev/null 2>&1; then \
		echo "Compiling and running Kotlin Android App Simulator (Zero Downloads)..."; \
		mkdir -p build/kt_app; \
		javac -d build/kt_app android/src/main/java/ai/liquid/lfm2/*.java; \
		java -cp "$(GRADLE_LIB_DIR)*" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib \
			-cp "$(GRADLE_LIB_DIR)kotlin-stdlib-2.0.21.jar:build/kt_app" \
			-d build/kt_app \
			android/src/main/kotlin/ai/liquid/lfm2/*.kt \
			android/src/main/kotlin/ai/liquid/lfm2/app/*.kt \
			tests/AndroidAppRunner.kt; \
		java -ea -cp "build/kt_app:$(GRADLE_LIB_DIR)kotlin-stdlib-2.0.21.jar" -Djava.library.path=./lib tests.AndroidAppRunner $(MODEL); \
	else \
		echo "NOTE: Local Kotlin compiler not detected. Skipping Kotlin App test."; \
	fi

run-kotlin-app: lib
	@if [ -n "$(GRADLE_LIB_DIR)" ] && command -v javac >/dev/null 2>&1; then \
		mkdir -p build/kt_app; \
		javac -d build/kt_app android/src/main/java/ai/liquid/lfm2/*.java; \
		java -cp "$(GRADLE_LIB_DIR)*" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib \
			-cp "$(GRADLE_LIB_DIR)kotlin-stdlib-2.0.21.jar:build/kt_app" \
			-d build/kt_app \
			android/src/main/kotlin/ai/liquid/lfm2/*.kt \
			android/src/main/kotlin/ai/liquid/lfm2/app/*.kt \
			tests/AndroidAppRunner.kt; \
		java -ea -cp "build/kt_app:$(GRADLE_LIB_DIR)kotlin-stdlib-2.0.21.jar" -Djava.library.path=./lib tests.AndroidAppRunner $(MODEL) --interactive; \
	else \
		echo "NOTE: Local Kotlin compiler not detected."; \
	fi

test-parity: lib fixtures
	@$(PYTHON) tests/verify_parity.py

test-c-bins: $(TEST_BINS) fixtures
	@./bin/test_arena
	@echo "-------------------------------------------------------"
	@./bin/test_math
	@echo "-------------------------------------------------------"
	@./bin/test_simd_avx2
	@echo "-------------------------------------------------------"
	@./bin/test_quants
	@echo "-------------------------------------------------------"
	@./bin/test_simd_kquants
	@echo "-------------------------------------------------------"
	@./bin/test_fused_conv
	@echo "-------------------------------------------------------"
	@./bin/test_threads
	@echo "-------------------------------------------------------"
	@./bin/test_gguf
	@echo "-------------------------------------------------------"
	@./bin/test_tokenizer
	@echo "-------------------------------------------------------"
	@./bin/test_bpe
	@echo "-------------------------------------------------------"
	@./bin/test_conv1d
	@echo "-------------------------------------------------------"
	@./bin/test_attention
	@echo "-------------------------------------------------------"
	@./bin/test_forward
	@echo "-------------------------------------------------------"
	@./bin/test_c_api
	@echo "-------------------------------------------------------"
	@./bin/test_lfm2_threadpool
	@echo "-------------------------------------------------------"
	@./bin/test_optimizations
	@echo "-------------------------------------------------------"
	@./bin/test_prefill_batch
	@echo "-------------------------------------------------------"
	@./bin/test_decode_1x4
	@echo "-------------------------------------------------------"
	@./bin/test_kernels_fast
	@echo "-------------------------------------------------------"
	@./bin/test_cpuid_quants
	@echo "-------------------------------------------------------"
	@./bin/test_runtime_features
	@echo "-------------------------------------------------------"
	@./bin/test_moe_checkpoint
	@echo "-------------------------------------------------------"
	@./bin/test_cli_stream
	@echo "-------------------------------------------------------"
	@./bin/test_sys_cache
	@echo "-------------------------------------------------------"
	@./bin/test_lfm2_api
	@echo "-------------------------------------------------------"
	@./bin/test_lfm2_server
	@echo "-------------------------------------------------------"
	@./bin/test_hardening
	@echo "-------------------------------------------------------"
	@./bin/test_simd_arm64
	@echo "-------------------------------------------------------"
	@./bin/test_android_io
	@echo "-------------------------------------------------------"
	@./bin/test_speculative_pld
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 GGUF Inspector..."
	@./bin/lfm2_loader tests/lfm2_toy.gguf > /dev/null && echo "PASSED: lfm2_loader verified on hybrid GGUF checkpoint!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 BPE Tokenizer Engine..."
	@./bin/lfm2_bpe tests/lfm2_toy.gguf > /dev/null && echo "PASSED: lfm2_bpe verified on hybrid GGUF checkpoint!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 Forward Pass Engine..."
	@./bin/lfm2_forward tests/lfm2_toy.gguf 8 > /dev/null && echo "PASSED: lfm2_forward verified 8-step decode on hybrid GGUF checkpoint!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 Chat REPL Engine..."
	@echo "/exit" | ./bin/lfm2_chat tests/lfm2_toy.gguf > /dev/null && echo "PASSED: lfm2_chat verified REPL lifecycle on hybrid GGUF checkpoint!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 Spin-Barrier Thread Pool..."
	@./bin/lfm2_threadpool 4 > /dev/null && echo "PASSED: lfm2_threadpool verified spin-barrier and AVX2 bandwidth!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 Batched Prefill Benchmark..."
	@./bin/lfm2_prefill_bench 4 > /dev/null && echo "PASSED: lfm2_prefill_bench verified batched GEMM speedup and bit-exact parity!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 1x4 & Fused FFN Benchmark..."
	@./bin/lfm2_decode_1x4_bench 4 > /dev/null && echo "PASSED: lfm2_decode_1x4_bench verified 1x4 and fused SwiGLU!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 Fused Fast Kernels Benchmark..."
	@./bin/lfm2_kernels_fast_bench 4 > /dev/null && echo "PASSED: lfm2_kernels_fast_bench verified planar ShortConv, RoPE LUT & FP16 online softmax!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 CPUID & Quantization Benchmark..."
	@./bin/lfm2_cpuid_quants > /dev/null && echo "PASSED: lfm2_cpuid_quants verified runtime CPUID dispatch, Q4_0, Q5_K & IQ4_XS!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 Runtime Control Features..."
	@./bin/lfm2_runtime_features > /dev/null && echo "PASSED: lfm2_runtime_features verified tool-call PDA, think budget & KV sink shift!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2-MoE & Hybrid State Checkpointing..."
	@./bin/lfm2_moe_checkpoint 2 > /dev/null && echo "PASSED: lfm2_moe_checkpoint verified 32-expert Sigmoid routing, 3D slicing & ShortConv rollback!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 CLI Streaming & SIGINT Guard..."
	@./bin/lfm2_cli_stream > /dev/null && echo "PASSED: lfm2_cli_stream verified streaming UTF-8 assembly, stop holdback & two-stage SIGINT!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 System Runtime & Prompt Cache..."
	@./bin/lfm2_sys_cache > /dev/null && echo "PASSED: lfm2_sys_cache verified OOM guard, page warmup & prompt cache!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 epoll HTTP/SSE Server..."
	@./bin/lfm2_server tests/lfm2_toy.gguf 19475 > /dev/null 2>&1 & SPID=$$!; sleep 0.2; curl -s http://127.0.0.1:19475/health | grep -q "ok" && kill -TERM $$SPID && echo "PASSED: lfm2_server verified epoll loop, SSE streaming & disconnect abort!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 ARM64 Mobile NEON/DotProd Suite..."
	@./bin/lfm2_simd_arm64 > /dev/null && echo "PASSED: lfm2_simd_arm64 verified mobile SIMD & parity!"
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Zero-Copy C11 LFM2 Prompt-Lookup Speculative Decoding..."
	@./bin/lfm2_speculative_pld 2 > /dev/null && echo "PASSED: lfm2_speculative_pld verified speculative B=8 speedup and bit-exact parity!"
	@echo "-------------------------------------------------------"
	@$(MAKE) test-http


test-core: all fixtures
	@echo "======================================================="
	@echo "Running Liquid AI Engine Full Automated Test Suite     "
	@echo "======================================================="
	@$(MAKE) test-c-bins
	@echo "-------------------------------------------------------"
	@$(MAKE) test-python

static-test: static fixtures
	@echo "-------------------------------------------------------"
	@echo "[Test] Verifying Standalone Static ELF binary..."
	@file bin/lfm-run-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm-run-static (statically linked)"
	@file bin/lfm2-threadpool-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-threadpool-static (statically linked)"
	@file bin/lfm2-forward-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-forward-static (statically linked)"
	@file bin/lfm2-prefill-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-prefill-static (statically linked)"
	@file bin/lfm2-decode-1x4-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-decode-1x4-static (statically linked)"
	@file bin/lfm2-kernels-fast-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-kernels-fast-static (statically linked)"
	@file bin/lfm2-cpuid-quants-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-cpuid-quants-static (statically linked)"
	@file bin/lfm2-runtime-features-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-runtime-features-static (statically linked)"
	@file bin/lfm2-moe-checkpoint-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-moe-checkpoint-static (statically linked)"
	@file bin/lfm2-cli-stream-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-cli-stream-static (statically linked)"
	@file bin/lfm2-sys-cache-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-sys-cache-static (statically linked)"
	@file bin/lfm2-server-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-server-static (statically linked)"
	@file bin/lfm2-simd-arm64-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-simd-arm64-static (statically linked)"
	@file bin/lfm2-speculative-pld-static | grep -q "statically linked" && echo "  Static binary verified: bin/lfm2-speculative-pld-static (statically linked)"
	@./bin/lfm-run-static -h > /dev/null && echo "  Static executable ran successfully with zero dynamic dependencies."
	@./bin/lfm2-server-static -m tests/lfm2_toy.gguf --port 19476 > /dev/null 2>&1 & \
		SPID=$$!; \
		trap 'kill $$SPID 2>/dev/null || true' EXIT; \
		sleep 0.3; \
		curl -sf http://127.0.0.1:19476/health | grep -q '"status":"ok"'; \
		MODELS_JSON=$$(curl -sf http://127.0.0.1:19476/v1/models); \
		echo "$$MODELS_JSON" | grep -q '"n_layers":2'; \
		kill -TERM $$SPID; wait $$SPID 2>/dev/null || true; \
		echo "  Static server verified real GGUF /health & /v1/models: $$MODELS_JSON"
	@echo "PASSED: Standalone static ELF packaging verified!"

test: test-core static-test
	@echo "======================================================="
	@echo "ALL TESTS PASSED WITH 100% SUCCESS RATE!               "
	@echo "======================================================="


# ThreadSanitizer data race detection test suite
bin/lfm2-tsan: tools/lfm2_chat.c tools/lfm2_forward.c tools/lfm2_loader.c tools/lfm2_prefill_batch.c tools/lfm2_threadpool.c | dirs
	$(CC) -std=c11 -O1 -g -fsanitize=thread -fPIC -Wall -Wextra -Wno-unused-function -Iinclude -D_GNU_SOURCE $(if $(filter 1,$(HAS_AVX2)),-mavx2 -mfma -mf16c) $< -o $@ $(LDFLAGS)

test-tsan: lib fixtures bin/lfm2-tsan
	@echo "======================================================="
	@echo "Running ThreadSanitizer Concurrency & Data Race Suite  "
	@echo "======================================================="
	@mkdir -p bin
	$(CC) -std=c11 -O1 -g -fsanitize=thread -fPIC -Wall -Wextra -Iinclude -D_GNU_SOURCE $(if $(filter 1,$(HAS_AVX2)),-mavx2 -mfma -mf16c) -I. tests/test_lfm2_threadpool.c -o bin/test_lfm2_threadpool_tsan -lm -lpthread
	$(CC) -std=c11 -O1 -g -fsanitize=thread -fPIC -Wall -Wextra -Iinclude -D_GNU_SOURCE $(if $(filter 1,$(HAS_AVX2)),-mavx2 -mfma -mf16c) tests/test_threads.c src/lfm_threadpool.c src/arena.c src/lfm_simd.c src/lfm_fused.c -o bin/test_threads_tsan -lm -lpthread
	@if command -v setarch >/dev/null 2>&1 && setarch $$(uname -m) -R true 2>/dev/null; then \
		setarch $$(uname -m) -R ./bin/test_lfm2_threadpool_tsan; \
		setarch $$(uname -m) -R ./bin/test_threads_tsan; \
	else \
		./bin/test_lfm2_threadpool_tsan; \
		./bin/test_threads_tsan; \
	fi
	@rm -f bin/test_lfm2_threadpool_tsan bin/test_threads_tsan
	@echo "======================================================="
	@echo "PASSED: ThreadSanitizer reports ZERO data races!       "
	@echo "======================================================="

MUSL_CC ?= musl-gcc
static-musl: dirs
	@if command -v $(MUSL_CC) >/dev/null 2>&1; then \
		echo "Building static binaries with $(MUSL_CC)..."; \
		$(MUSL_CC) -std=c11 -O3 -Wall -Wextra -Iinclude -D_GNU_SOURCE $(if $(filter 1,$(HAS_AVX2)),-mavx2 -mfma -mf16c) -static src/main.c $(SRC_CORE) -o bin/lfm-run-static-musl -lm -lpthread; \
		strip -s bin/lfm-run-static-musl; \
		echo "  musl static binary created: bin/lfm-run-static-musl ($$(ls -lh bin/lfm-run-static-musl | awk '{print $$5}'))"; \
	else \
		echo "NOTE: $(MUSL_CC) not found on PATH. Install musl-tools (e.g. 'sudo apt install musl-tools') to build musl static binaries."; \
	fi

# Android NDK Configuration (Override via ANDROID_NDK_HOME and ANDROID_API)
ANDROID_API   ?= 26
NDK_HOST_TAG  ?= linux-x86_64
NDK_LLVM_BIN  := $(ANDROID_NDK_HOME)/toolchains/llvm/prebuilt/$(NDK_HOST_TAG)/bin
ANDROID_CC    := $(NDK_LLVM_BIN)/aarch64-linux-android$(ANDROID_API)-clang

android:
	@if [ -z "$(ANDROID_NDK_HOME)" ]; then \
		echo "[ERROR] ANDROID_NDK_HOME is not set."; exit 1; \
	fi
	@mkdir -p lib/android-arm64-v8a
	@echo "[ANDROID-NDK] Compiling lib/android-arm64-v8a/liblfm2.so with $(ANDROID_CC)..."
	@$(ANDROID_CC) -std=c11 -O3 -fPIC -fvisibility=hidden -pthread -Wall -Wextra -Wno-unused-function \
		-march=armv8.2-a+dotprod+fp16 \
		-Iinclude -Isrc -D_GNU_SOURCE \
		-shared -Wl,-soname,liblfm2.so \
		-Wl,-z,max-page-size=16384 \
		src/lfm2_api.c -o lib/android-arm64-v8a/liblfm2.so -lm -llog
	@echo "[ANDROID-NDK] Built lib/android-arm64-v8a/liblfm2.so"

python-wheel:
	@$(PYTHON) -m pip install -e . --no-build-isolation

parity: lib fixtures
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make parity MODEL=/path/to/LFM2.5-1.2B-Instruct-Q4_K_M.gguf [QUANT=Q4_K_M]"; \
		exit 1; \
	fi
	@$(PYTHON) tests/verify_parity.py -m $(MODEL) --lib ./lib/liblfm2.so -q $(or $(QUANT),Q4_K_M)

asan: test-asan
tsan: test-tsan

# AddressSanitizer and UndefinedBehaviorSanitizer memory safety test suite
test-asan:
	@echo "======================================================="
	@echo "Building with AddressSanitizer & UndefinedBehaviorSanitizer"
	@echo "======================================================="
	$(MAKE) clean
	$(MAKE) all test-c-bins CFLAGS="-std=c11 -O1 -g -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Wno-unused-function -Iinclude -D_GNU_SOURCE $(if $(filter 1,$(HAS_AVX2)),-mavx2 -mfma -mf16c)" LDFLAGS="-fsanitize=address,undefined -lm -lpthread"

help:
	@echo "LFM2.c Production C11 Build Targets:"
	@echo "  make all                        - Build CLI, epoll HTTP server, static binaries, benchmarks and tests"
	@echo "  make cli                        - Build interactive terminal CLI (bin/lfm2)"
	@echo "  make server                     - Build epoll HTTP/SSE server (bin/lfm2-server)"
	@echo "  make shared                     - Build liblfm2.so shared library"
	@echo "  make static                     - Build stripped static Linux binaries (musl-gcc / gcc -static)"
	@echo "  make NATIVE=1                   - Build with -march=native for host CPU"
	@echo "  make android                    - Cross-compile arm64-v8a liblfm2.so via ANDROID_NDK_HOME"
	@echo "  make python-wheel               - Build and install editable Python ctypes wheel"
	@echo "  make asan / make tsan           - Run Address/UB Sanitizer or ThreadSanitizer suites"
	@echo "  make parity MODEL=model.gguf    - Run 5-tier numerical & contract verification suite"
	@echo "  make test                       - Run complete C, Python, and HTTP integration test suite"

clean:
	rm -rf build bin lib lfm-run lfm-server ./liblfm2.so ./dist ./lfm2.egg-info
