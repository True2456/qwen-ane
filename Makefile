# SPDX-License-Identifier: Apache-2.0
# Rindi - Apple Silicon Standalone Native C++ ANE + Metal GPU Inference Server

CC ?= clang
CXX ?= clang++

CFLAGS = -O3 -Wall -fPIC -Wno-deprecated-declarations
CXXFLAGS = -O3 -std=c++17 -Wall -fPIC -Wno-deprecated-declarations
SME2FLAGS = -march=armv9.2-a+sme2 -fno-vectorize -fno-slp-vectorize
FRAMEWORKS = -framework Foundation -framework Metal -framework IOSurface -framework CoreGraphics

RUNTIME_DIR = runtime

OBJS = \
	$(RUNTIME_DIR)/ane_c_bridge.o \
	$(RUNTIME_DIR)/metal_engine.o \
	$(RUNTIME_DIR)/rindi_native_chain.o \
	$(RUNTIME_DIR)/rindi_ane_projection.o \
	$(RUNTIME_DIR)/rindi_gdn_state.o \
	$(RUNTIME_DIR)/rindi_gdn_layer.o \
	$(RUNTIME_DIR)/rindi_gdn_conv.o \
	$(RUNTIME_DIR)/rindi_gdn_recurrence.o \
	$(RUNTIME_DIR)/rindi_attention.o \
	$(RUNTIME_DIR)/rindi_sme_engine.o \
	$(RUNTIME_DIR)/bpe_tokenizer.o \
	$(RUNTIME_DIR)/rindi_engine.o \
	$(RUNTIME_DIR)/rindi_tui.o \
	$(RUNTIME_DIR)/rindi_mtp.o \
	$(RUNTIME_DIR)/rindi_c_api.o

TARGETS = \
	$(RUNTIME_DIR)/libmetal_engine.dylib \
	$(RUNTIME_DIR)/librindi_native.dylib \
	$(RUNTIME_DIR)/rindi-server \
	bin/ane-as \
	runtime/librindi_ane_swift.dylib \
	test-coreai-tail

.PHONY: all clean test-ane-as test-cpp-safetensors test-ane-bridge test-ane-projection test-ane-int4-projection test-gdn-state test-gdn-conv test-gdn-recurrence test-gdn-layer test-attention test-metal-attention test-mtp test-engine-sampling test-prefill-mm test-native-generate test-ane-multi-output test-ane-artifact-load test-coreai-tail test-sme2 bench-sme2 bench-sme2-hetero bench-sme2-dense-decode bench-native-mtp bench-native-ane-wide eval-native-ppl

all: $(TARGETS)

bin/ane-as: tools/ane_as.cpp $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -I. $< $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -o /tmp/rindi-ane-as
	@cp /tmp/rindi-ane-as $@

test-ane-as: bin/ane-as
	/tmp/rindi-ane-as --iters 50

/tmp/rindi-test-prefill-mm: probes/test_ane_prefill_mm.cpp $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -o $@

test-prefill-mm: /tmp/rindi-test-prefill-mm
	/tmp/rindi-test-prefill-mm

test-gemm-simd-exact: probes/test_gemm_simd_exact.cpp $(RUNTIME_DIR)/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< $(RUNTIME_DIR)/metal_engine.o -framework Metal -framework Foundation -framework IOSurface -o /tmp/rindi-gemm-simd-exact
	/tmp/rindi-gemm-simd-exact

/tmp/rindi-test-sme2: probes/test_sme2_q4.cpp $(RUNTIME_DIR)/rindi_sme_engine.o
	$(CXX) $(CXXFLAGS) -I. $^ -o $@

test-sme2: /tmp/rindi-test-sme2
	/tmp/rindi-test-sme2

/tmp/rindi-bench-sme2: probes/bench_sme2_q4.cpp $(RUNTIME_DIR)/rindi_sme_engine.o $(RUNTIME_DIR)/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $^ $(FRAMEWORKS) -o $@

bench-sme2: /tmp/rindi-bench-sme2
	/tmp/rindi-bench-sme2

/tmp/rindi-bench-sme2-hetero: probes/bench_hetero_q4.cpp $(RUNTIME_DIR)/rindi_sme_engine.o $(RUNTIME_DIR)/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $^ $(FRAMEWORKS) -o $@

bench-sme2-hetero: /tmp/rindi-bench-sme2-hetero
	/tmp/rindi-bench-sme2-hetero dense_down 100 4
	/tmp/rindi-bench-sme2-hetero dense_gate_up 100 4

/tmp/rindi-bench-sme2-dense-decode: probes/bench_sme2_dense_decode.cpp $(OBJS) runtime/librindi_ane_swift.dylib
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -o $@

bench-sme2-dense-decode: /tmp/rindi-bench-sme2-dense-decode
	RINDI_TAIL_COREAI=1 RINDI_ENABLE_METAL_TAIL=1 $< $(HOME)/.lmstudio/models/Qwen/Qwen3.8-27B.rindi 12
	RINDI_TAIL_COREAI=1 RINDI_SME2_DOWN=1 RINDI_SME2_WORKERS=4 $< $(HOME)/.lmstudio/models/Qwen/Qwen3.8-27B.rindi 12

/tmp/rindi-bench-native-mtp: probes/bench_native_mtp.cpp $(OBJS) runtime/librindi_ane_swift.dylib
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o $@

/tmp/rindi-eval-native-ppl: probes/eval_native_ppl.cpp $(OBJS) runtime/librindi_ane_swift.dylib
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o $@

eval-native-ppl: /tmp/rindi-eval-native-ppl
	RINDI_DISABLE_QWEN_PREFILL_FAST=1 $< $(HOME)/.lmstudio/models/Qwen/Qwen3.8-27B.rindi /tmp/rindi-wikitext-validation.txt 256
	RINDI_QWEN_PREFILL_FAST=1 $< $(HOME)/.lmstudio/models/Qwen/Qwen3.8-27B.rindi /tmp/rindi-wikitext-validation.txt 256

bench-native-mtp: /tmp/rindi-bench-native-mtp
	RINDI_DISABLE_MTP=1 RINDI_TAIL_COREAI=1 RINDI_ENABLE_METAL_TAIL=1 RINDI_PREFILL_BATCH_ATTENTION=1 $< $(HOME)/.lmstudio/models/Qwen/Qwen3.8-27B.rindi 1024 128
	env -u RINDI_DISABLE_MTP RINDI_TAIL_COREAI=1 RINDI_ENABLE_METAL_TAIL=1 RINDI_PREFILL_BATCH_ATTENTION=1 RINDI_STATE_REBUILD=1 $< $(HOME)/.lmstudio/models/Qwen/Qwen3.8-27B.rindi 1024 128

bench-native-ane-wide: /tmp/rindi-bench-native-mtp
	RINDI_DISABLE_MTP=1 RINDI_TAIL_COREAI=1 RINDI_ENABLE_METAL_TAIL=1 RINDI_PREFILL_BATCH_ATTENTION=1 RINDI_ANE_WIDTH=128 $< $(HOME)/.lmstudio/models/Qwen/Qwen3.8-27B.rindi 1024 128

test-cpp-safetensors: probes/test_cpp_safetensors.cpp runtime/safetensors_loader.h
	$(CXX) $(CXXFLAGS) -I. $< -o /tmp/rindi-test-cpp-safetensors
	/tmp/rindi-test-cpp-safetensors

test-ane-bridge: probes/test_ane_bridge.cpp runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -o /tmp/rindi-test-ane-bridge
	/tmp/rindi-test-ane-bridge

test-ane-projection: probes/test_ane_projection.cpp runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-ane-projection
	/tmp/rindi-test-ane-projection

test-ane-int4-projection: probes/test_ane_int4_projection.cpp runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-ane-int4-projection
	/tmp/rindi-test-ane-int4-projection

test-gdn-state: probes/test_gdn_state.cpp runtime/rindi_gdn_state.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_gdn_state.o -o /tmp/rindi-test-gdn-state
	/tmp/rindi-test-gdn-state

test-gdn-conv: probes/test_gdn_conv.cpp runtime/rindi_gdn_conv.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_gdn_conv.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -o /tmp/rindi-test-gdn-conv
	/tmp/rindi-test-gdn-conv

test-gdn-recurrence: probes/test_gdn_recurrence.cpp runtime/rindi_gdn_recurrence.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_gdn_recurrence.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -o /tmp/rindi-test-gdn-recurrence
	/tmp/rindi-test-gdn-recurrence

test-gdn-layer: probes/test_gdn_layer.cpp runtime/rindi_gdn_layer.o runtime/rindi_gdn_conv.o runtime/rindi_gdn_recurrence.o runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_gdn_layer.o runtime/rindi_gdn_conv.o runtime/rindi_gdn_recurrence.o runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-gdn-layer
	RINDI_TAIL_COREAI=1 RINDI_QWEN_PREFILL_FAST=1 /tmp/rindi-test-gdn-layer 32 29

test-attention: probes/test_attention.cpp runtime/rindi_attention.o runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_attention.o runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-attention
	/tmp/rindi-test-attention

test-mtp: probes/test_mtp.cpp $(OBJS) runtime/librindi_ane_swift.dylib
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-mtp
	RINDI_TAIL_COREAI=1 RINDI_ENABLE_METAL_TAIL=1 /tmp/rindi-test-mtp

test-metal-attention: probes/test_metal_attention.cpp runtime/rindi_attention.o runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_attention.o runtime/rindi_ane_projection.o runtime/rindi_sme_engine.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-metal-attention
	/tmp/rindi-test-metal-attention

test-engine-sampling: probes/test_engine_sampling.cpp $(OBJS) runtime/librindi_ane_swift.dylib
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-engine-sampling
	/tmp/rindi-test-engine-sampling

test-native-generate: probes/test_native_generate.cpp $(OBJS) runtime/librindi_ane_swift.dylib
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-native-generate
	/tmp/rindi-test-native-generate

test-ane-multi-output: probes/test_ane_multi_output.cpp runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -o /tmp/rindi-test-ane-multi-output
	/tmp/rindi-test-ane-multi-output

test-ane-artifact-load: probes/test_ane_artifact_load.cpp $(RUNTIME_DIR)/rindi_native_chain.o $(RUNTIME_DIR)/rindi_mtp.o $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< $(RUNTIME_DIR)/rindi_native_chain.o $(RUNTIME_DIR)/rindi_mtp.o $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -o /tmp/rindi-test-ane-artifact-load
	/tmp/rindi-test-ane-artifact-load

$(RUNTIME_DIR)/ane_c_bridge.o: $(RUNTIME_DIR)/ane_c_bridge.m $(RUNTIME_DIR)/ane_c_bridge.h
	$(CC) $(CFLAGS) -c $< -o $@

$(RUNTIME_DIR)/metal_engine.o: $(RUNTIME_DIR)/metal_engine.m $(RUNTIME_DIR)/metal_engine.h
	$(CC) $(CFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_native_chain.o: $(RUNTIME_DIR)/rindi_native_chain.cpp $(RUNTIME_DIR)/rindi_native_chain.h $(RUNTIME_DIR)/rindi_ane_projection.h $(RUNTIME_DIR)/metal_engine.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_ane_projection.o: $(RUNTIME_DIR)/rindi_ane_projection.cpp $(RUNTIME_DIR)/rindi_ane_projection.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_gdn_state.o: $(RUNTIME_DIR)/rindi_gdn_state.cpp $(RUNTIME_DIR)/rindi_gdn_state.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_gdn_layer.o: $(RUNTIME_DIR)/rindi_gdn_layer.cpp $(RUNTIME_DIR)/rindi_gdn_layer.h $(RUNTIME_DIR)/rindi_ane_projection.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_gdn_conv.o: $(RUNTIME_DIR)/rindi_gdn_conv.cpp $(RUNTIME_DIR)/rindi_gdn_conv.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_gdn_recurrence.o: $(RUNTIME_DIR)/rindi_gdn_recurrence.cpp $(RUNTIME_DIR)/rindi_gdn_recurrence.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_attention.o: $(RUNTIME_DIR)/rindi_attention.cpp $(RUNTIME_DIR)/rindi_attention.h $(RUNTIME_DIR)/rindi_ane_projection.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_sme_engine.o: $(RUNTIME_DIR)/rindi_sme_engine.cpp $(RUNTIME_DIR)/rindi_sme_engine.h
	$(CXX) $(CXXFLAGS) $(SME2FLAGS) -c $< -o $@

$(RUNTIME_DIR)/bpe_tokenizer.o: $(RUNTIME_DIR)/bpe_tokenizer.cpp $(RUNTIME_DIR)/bpe_tokenizer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_engine.o: $(RUNTIME_DIR)/rindi_engine.cpp $(RUNTIME_DIR)/rindi_engine.h $(RUNTIME_DIR)/safetensors_loader.h $(RUNTIME_DIR)/rindi_ane_projection.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_tui.o: $(RUNTIME_DIR)/rindi_tui.cpp $(RUNTIME_DIR)/rindi_tui.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_mtp.o: $(RUNTIME_DIR)/rindi_mtp.cpp $(RUNTIME_DIR)/rindi_mtp.h $(RUNTIME_DIR)/rindi_attention.h $(RUNTIME_DIR)/rindi_ane_projection.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_c_api.o: $(RUNTIME_DIR)/rindi_c_api.cpp $(RUNTIME_DIR)/rindi_c_api.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_server.o: $(RUNTIME_DIR)/rindi_server.cpp $(RUNTIME_DIR)/rindi_engine.h $(RUNTIME_DIR)/rindi_native_chain.h $(RUNTIME_DIR)/rindi_tui.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/libmetal_engine.dylib: $(RUNTIME_DIR)/metal_engine.o
	$(CC) -dynamiclib -O3 $^ $(FRAMEWORKS) -o $@

$(RUNTIME_DIR)/librindi_native.dylib: $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o $(RUNTIME_DIR)/rindi_native_chain.o $(RUNTIME_DIR)/rindi_ane_projection.o $(RUNTIME_DIR)/rindi_gdn_state.o $(RUNTIME_DIR)/rindi_gdn_layer.o $(RUNTIME_DIR)/rindi_gdn_conv.o $(RUNTIME_DIR)/rindi_gdn_recurrence.o $(RUNTIME_DIR)/rindi_attention.o $(RUNTIME_DIR)/rindi_sme_engine.o $(RUNTIME_DIR)/rindi_mtp.o $(RUNTIME_DIR)/bpe_tokenizer.o $(RUNTIME_DIR)/rindi_engine.o $(RUNTIME_DIR)/rindi_c_api.o runtime/librindi_ane_swift.dylib
	$(CXX) -dynamiclib -O3 $^ $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -Wl,-rpath,@loader_path -o $@

$(RUNTIME_DIR)/rindi-server: $(OBJS) $(RUNTIME_DIR)/rindi_server.o runtime/librindi_ane_swift.dylib
	$(CXX) -O3 $^ $(FRAMEWORKS) -Lruntime -lrindi_ane_swift -Wl,-rpath,@executable_path -Wl,-rpath,@loader_path -o $@

clean:
	rm -f $(RUNTIME_DIR)/*.o $(TARGETS)

runtime/librindi_ane_swift.dylib: runtime/rindi_ane_swift.swift
	swiftc -c -parse-as-library -emit-library -O -Xlinker -install_name -Xlinker @rpath/librindi_ane_swift.dylib -o $@ $<

test-coreai-tail: probes/test_coreai_tail.cpp $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o runtime/librindi_ane_swift.dylib
	$(CXX) $(CXXFLAGS) -I. $< $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o $(FRAMEWORKS) -Lruntime -lrindi_ane_swift '-Wl,-rpath,$(abspath runtime)' -o /tmp/rindi-test-coreai-tail
