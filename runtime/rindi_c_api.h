/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_c_api.h - C API Export for Rindi Native Engine.
 */

#ifndef RINDI_C_API_H
#define RINDI_C_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RindiNativeEngineHandle RindiNativeEngineHandle;

/**
 * Create a native 64-layer ANE engine handle.
 */
RindiNativeEngineHandle* rindi_engine_create(size_t hidden_dim, size_t seq_len);

/**
 * Load a precompiled ANE layer into the engine.
 */
bool rindi_engine_load_layer(RindiNativeEngineHandle* handle, int layer_idx, const char* package_path);

/**
 * Run a full 64-layer forward pass on ANE in pure C.
 */
bool rindi_engine_evaluate_step(RindiNativeEngineHandle* handle, const void* input_fp16, void* output_fp16);

/**
 * Get last evaluation latency in milliseconds.
 */
double rindi_engine_get_last_latency_ms(RindiNativeEngineHandle* handle);

/**
 * Destroy engine handle and free all resources.
 */
void rindi_engine_destroy(RindiNativeEngineHandle* handle);

#ifdef __cplusplus
}
#endif

#endif // RINDI_C_API_H
