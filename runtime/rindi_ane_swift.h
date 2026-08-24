/*
 * runtime/rindi_ane_swift.h - C ABI for the CoreAI Swift shim.
 *
 * Loads coreai-torch-exported .aimodel bundles (INT4-palettized GDN tails)
 * through Apple's official CoreAI runtime with persistent specialization
 * cache, running on CPU/GPU/ANE per preference. This is the macOS 27 ANE
 * backend; the legacy MIL private-API path is dead on 27 (bundle-format
 * verification rejects legacy MIL).
 */
#ifndef RINDI_ANE_SWIFT_H
#define RINDI_ANE_SWIFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Load a .aimodel bundle. preferANE: 1 = Neural Engine, 0 = CPU.
 * Returns opaque handle or NULL (rindi_swift_last_error() has details). */
void* rindi_ane_load(const char* bundle_path, int prefer_ane);

/* Evaluate one function invocation. Inputs/outputs are fp16 bit patterns.
 * xin layout: row-major [rows x cols] (token-major activations).
 * outs: concatenated outputs in function output order, caller-allocated;
 * pass out_cap=0 with outs=NULL to query total element count.
 * Returns total elements written (>0), or negative on error. */
long rindi_ane_run(void* handle, const uint16_t* xin, long rows, long cols,
                   uint16_t* outs, long out_cap);

void rindi_ane_free(void* handle);

const char* rindi_ane_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* RINDI_ANE_SWIFT_H */
