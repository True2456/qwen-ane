// SPDX-License-Identifier: Apache-2.0
// Minimal C ABI for embedding the SME2 projection backend.

#include "rindi_sme_engine.h"

#include <cstddef>
#include <cstdint>
#include <new>

extern "C" {

void* rindi_sme_create(size_t workers) {
    auto* engine = new (std::nothrow) RindiSmeEngine(workers);
    if (!engine || !engine->is_available()) {
        delete engine;
        return nullptr;
    }
    return engine;
}

void rindi_sme_destroy(void* opaque) {
    delete static_cast<RindiSmeEngine*>(opaque);
}

int rindi_sme_q8_fp16(void* opaque, const uint16_t* x,
                      const int8_t* weights, const uint16_t* scales,
                      uint16_t* output, size_t rows, size_t cols) {
    if (!opaque) return 0;
    return static_cast<RindiSmeEngine*>(opaque)->gemv_q8_rowwise_fp16(
        x, weights, scales, output, rows, cols) ? 1 : 0;
}

}  // extern "C"
