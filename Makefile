# SPDX-License-Identifier: Apache-2.0
# Rindi - Apple Silicon Standalone Native C++ ANE + Metal GPU Inference Server

CC ?= clang
CXX ?= clang++

CFLAGS = -O3 -Wall -fPIC -Wno-deprecated-declarations
CXXFLAGS = -O3 -std=c++17 -Wall -fPIC -Wno-deprecated-declarations
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
	$(RUNTIME_DIR)/bpe_tokenizer.o \
	$(RUNTIME_DIR)/rindi_engine.o \
	$(RUNTIME_DIR)/rindi_tui.o \
	$(RUNTIME_DIR)/rindi_mtp.o \
	$(RUNTIME_DIR)/rindi_c_api.o

TARGETS = \
	$(RUNTIME_DIR)/libmetal_engine.dylib \
	$(RUNTIME_DIR)/librindi_native.dylib \
	$(RUNTIME_DIR)/rindi-server

.PHONY: all clean test-cpp-safetensors test-ane-bridge test-ane-projection test-ane-int4-projection test-gdn-state test-gdn-conv test-gdn-recurrence test-gdn-layer test-attention test-metal-attention test-mtp test-engine-sampling test-native-generate test-ane-multi-output test-ane-artifact-load

all: $(TARGETS)

test-cpp-safetensors: probes/test_cpp_safetensors.cpp runtime/safetensors_loader.h
	$(CXX) $(CXXFLAGS) -I. $< -o /tmp/rindi-test-cpp-safetensors
	/tmp/rindi-test-cpp-safetensors

test-ane-bridge: probes/test_ane_bridge.cpp runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-ane-bridge
	/tmp/rindi-test-ane-bridge

test-ane-projection: probes/test_ane_projection.cpp runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-ane-projection
	/tmp/rindi-test-ane-projection

test-ane-int4-projection: probes/test_ane_int4_projection.cpp runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-ane-int4-projection
	/tmp/rindi-test-ane-int4-projection

test-gdn-state: probes/test_gdn_state.cpp runtime/rindi_gdn_state.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_gdn_state.o -o /tmp/rindi-test-gdn-state
	/tmp/rindi-test-gdn-state

test-gdn-conv: probes/test_gdn_conv.cpp runtime/rindi_gdn_conv.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_gdn_conv.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-gdn-conv
	/tmp/rindi-test-gdn-conv

test-gdn-recurrence: probes/test_gdn_recurrence.cpp runtime/rindi_gdn_recurrence.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_gdn_recurrence.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-gdn-recurrence
	/tmp/rindi-test-gdn-recurrence

test-gdn-layer: probes/test_gdn_layer.cpp runtime/rindi_gdn_layer.o runtime/rindi_gdn_conv.o runtime/rindi_gdn_recurrence.o runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_gdn_layer.o runtime/rindi_gdn_conv.o runtime/rindi_gdn_recurrence.o runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-gdn-layer
	/tmp/rindi-test-gdn-layer

test-attention: probes/test_attention.cpp runtime/rindi_attention.o runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_attention.o runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-attention
	/tmp/rindi-test-attention

test-mtp: probes/test_mtp.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -o /tmp/rindi-test-mtp
	/tmp/rindi-test-mtp

test-metal-attention: probes/test_metal_attention.cpp runtime/rindi_attention.o runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/rindi_attention.o runtime/rindi_ane_projection.o runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-metal-attention
	/tmp/rindi-test-metal-attention

test-engine-sampling: probes/test_engine_sampling.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -o /tmp/rindi-test-engine-sampling
	/tmp/rindi-test-engine-sampling

test-native-generate: probes/test_native_generate.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -I. $< $(OBJS) $(FRAMEWORKS) -o /tmp/rindi-test-native-generate
	/tmp/rindi-test-native-generate

test-ane-multi-output: probes/test_ane_multi_output.cpp runtime/ane_c_bridge.o runtime/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< runtime/ane_c_bridge.o runtime/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-ane-multi-output
	/tmp/rindi-test-ane-multi-output

test-ane-artifact-load: probes/test_ane_artifact_load.cpp $(RUNTIME_DIR)/rindi_native_chain.o $(RUNTIME_DIR)/rindi_mtp.o $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o
	$(CXX) $(CXXFLAGS) -I. $< $(RUNTIME_DIR)/rindi_native_chain.o $(RUNTIME_DIR)/rindi_mtp.o $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o $(FRAMEWORKS) -o /tmp/rindi-test-ane-artifact-load
	/tmp/rindi-test-ane-artifact-load

$(RUNTIME_DIR)/ane_c_bridge.o: $(RUNTIME_DIR)/ane_c_bridge.m $(RUNTIME_DIR)/ane_c_bridge.h
	$(CC) $(CFLAGS) -c $< -o $@

$(RUNTIME_DIR)/metal_engine.o: $(RUNTIME_DIR)/metal_engine.m $(RUNTIME_DIR)/metal_engine.h
	$(CC) $(CFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_native_chain.o: $(RUNTIME_DIR)/rindi_native_chain.cpp $(RUNTIME_DIR)/rindi_native_chain.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_ane_projection.o: $(RUNTIME_DIR)/rindi_ane_projection.cpp $(RUNTIME_DIR)/rindi_ane_projection.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_gdn_state.o: $(RUNTIME_DIR)/rindi_gdn_state.cpp $(RUNTIME_DIR)/rindi_gdn_state.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_gdn_layer.o: $(RUNTIME_DIR)/rindi_gdn_layer.cpp $(RUNTIME_DIR)/rindi_gdn_layer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_gdn_conv.o: $(RUNTIME_DIR)/rindi_gdn_conv.cpp $(RUNTIME_DIR)/rindi_gdn_conv.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_gdn_recurrence.o: $(RUNTIME_DIR)/rindi_gdn_recurrence.cpp $(RUNTIME_DIR)/rindi_gdn_recurrence.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_attention.o: $(RUNTIME_DIR)/rindi_attention.cpp $(RUNTIME_DIR)/rindi_attention.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/bpe_tokenizer.o: $(RUNTIME_DIR)/bpe_tokenizer.cpp $(RUNTIME_DIR)/bpe_tokenizer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_engine.o: $(RUNTIME_DIR)/rindi_engine.cpp $(RUNTIME_DIR)/rindi_engine.h $(RUNTIME_DIR)/safetensors_loader.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_tui.o: $(RUNTIME_DIR)/rindi_tui.cpp $(RUNTIME_DIR)/rindi_tui.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_mtp.o: $(RUNTIME_DIR)/rindi_mtp.cpp $(RUNTIME_DIR)/rindi_mtp.h $(RUNTIME_DIR)/rindi_attention.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_c_api.o: $(RUNTIME_DIR)/rindi_c_api.cpp $(RUNTIME_DIR)/rindi_c_api.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/rindi_server.o: $(RUNTIME_DIR)/rindi_server.cpp $(RUNTIME_DIR)/rindi_engine.h $(RUNTIME_DIR)/rindi_native_chain.h $(RUNTIME_DIR)/rindi_tui.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_DIR)/libmetal_engine.dylib: $(RUNTIME_DIR)/metal_engine.o
	$(CC) -dynamiclib -O3 $^ $(FRAMEWORKS) -o $@

$(RUNTIME_DIR)/librindi_native.dylib: $(RUNTIME_DIR)/ane_c_bridge.o $(RUNTIME_DIR)/metal_engine.o $(RUNTIME_DIR)/rindi_native_chain.o $(RUNTIME_DIR)/rindi_ane_projection.o $(RUNTIME_DIR)/rindi_gdn_state.o $(RUNTIME_DIR)/rindi_gdn_layer.o $(RUNTIME_DIR)/rindi_gdn_conv.o $(RUNTIME_DIR)/rindi_gdn_recurrence.o $(RUNTIME_DIR)/rindi_attention.o $(RUNTIME_DIR)/rindi_mtp.o $(RUNTIME_DIR)/bpe_tokenizer.o $(RUNTIME_DIR)/rindi_engine.o $(RUNTIME_DIR)/rindi_c_api.o
	$(CXX) -dynamiclib -O3 $^ $(FRAMEWORKS) -o $@

$(RUNTIME_DIR)/rindi-server: $(OBJS) $(RUNTIME_DIR)/rindi_server.o
	$(CXX) -O3 $^ $(FRAMEWORKS) -o $@

clean:
	rm -f $(RUNTIME_DIR)/*.o $(TARGETS)
