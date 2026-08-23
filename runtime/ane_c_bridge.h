/*
 * SPDX-License-Identifier: Apache-2.0
 * ane_c_bridge.h - Pure C interface for private AppleNeuralEngine.framework.
 */

#ifndef ANE_C_BRIDGE_H
#define ANE_C_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <IOSurface/IOSurfaceRef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANEContext ANEContext;
typedef struct ANEModel ANEModel;
typedef struct ANERequest ANERequest;

/**
 * Initialize ANE Context and client connection.
 */
ANEContext* ane_context_create(void);

/**
 * Destroy ANE Context.
 */
void ane_context_destroy(ANEContext* ctx);

/**
 * Load a precompiled ANE model directory (.hwx / compiled package).
 */
ANEModel* ane_model_load_compiled(ANEContext* ctx, const char* package_path, const char* key, int qos);

/**
 * Build/load an ANE model from MIL text and raw weight payloads. Payloads are
 * wrapped in the milinternal 128-byte blob format used by q38_ane_engine.py.
 * `weight_names` must match the @model_path/weights/<name> paths in MIL.
 */
ANEModel* ane_model_compile_mil(
    ANEContext* ctx,
    const char* mil_text,
    const char* const* weight_names,
    const void* const* weight_data,
    const size_t* weight_sizes,
    size_t weight_count,
    int instance_hint,
    int qos
);

/**
 * Return the channel dimension of each compiled output, in the ANE symbol
 * order.  MIL may reorder tuple outputs during compilation, so callers that
 * bind multiple IOSurfaces must use this order rather than the source tuple
 * order.
 */
size_t ane_model_output_channels(
    ANEModel* model,
    size_t* channels,
    size_t capacity
);

/**
 * Free an ANEModel.
 */
void ane_model_release(ANEModel* model);

/**
 * Create an evaluation request binding input and output IOSurfaces.
 */
ANERequest* ane_request_create(
    ANEContext* ctx,
    ANEModel* model,
    IOSurfaceRef input_surface,
    IOSurfaceRef output_surface,
    int procedure_index
);

/**
 * Create an evaluation request binding TWO input IOSurfaces (arg order maps
 * to the MIL function's first two parameters) and one output.
 */
ANERequest* ane_request_create_2in(
    ANEContext* ctx,
    ANEModel* model,
    IOSurfaceRef input1,
    IOSurfaceRef input2,
    IOSurfaceRef output_surface,
    int procedure_index
);

/**
 * Evaluate via the real-time client path (evaluateRealTimeWithModel:) -
 * lower-latency scheduling variant; same request/buffers as the direct path.
 */
bool ane_request_evaluate_realtime(
    ANEContext* ctx,
    ANEModel* model,
    ANERequest* req
);

ANERequest* ane_request_create_multi(
    ANEContext* ctx,
    ANEModel* model,
    IOSurfaceRef input_surface,
    IOSurfaceRef* output_surfaces,
    size_t output_count,
    int procedure_index
);

/**
 * Free an ANERequest.
 */
void ane_request_release(ANERequest* req);

/**
 * Evaluate an ANERequest on hardware with optional Metal SharedEvent signaling.
 */
bool ane_request_evaluate(
    ANEContext* ctx,
    ANEModel* model,
    ANERequest* req,
    void* wait_shared_event,
    uint64_t wait_value,
    void* signal_shared_event,
    uint64_t signal_value
);

#ifdef __cplusplus
}
#endif

#endif /* ANE_C_BRIDGE_H */
