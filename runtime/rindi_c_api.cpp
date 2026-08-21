/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_c_api.cpp - C API Implementation for Rindi Native Engine.
 */

#include "rindi_c_api.h"
#include "rindi_native_chain.h"

struct RindiNativeEngineHandle {
    RindiNativeChain* chain;
};

extern "C" {

RindiNativeEngineHandle* rindi_engine_create(size_t hidden_dim, size_t seq_len) {
    auto handle = new RindiNativeEngineHandle();
    handle->chain = new RindiNativeChain(hidden_dim, seq_len);
    return handle;
}

bool rindi_engine_load_layer(RindiNativeEngineHandle* handle, int layer_idx, const char* package_path) {
    if (!handle || !handle->chain || !package_path) return false;
    return handle->chain->load_layer(layer_idx, std::string(package_path));
}

bool rindi_engine_evaluate_step(RindiNativeEngineHandle* handle, const void* input_fp16, void* output_fp16) {
    if (!handle || !handle->chain) return false;
    return handle->chain->evaluate_step(input_fp16, output_fp16);
}

double rindi_engine_get_last_latency_ms(RindiNativeEngineHandle* handle) {
    if (!handle || !handle->chain) return 0.0;
    return handle->chain->get_last_eval_ms();
}

void rindi_engine_destroy(RindiNativeEngineHandle* handle) {
    if (handle) {
        delete handle->chain;
        delete handle;
    }
}

}
