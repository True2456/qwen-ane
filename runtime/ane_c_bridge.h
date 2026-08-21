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
