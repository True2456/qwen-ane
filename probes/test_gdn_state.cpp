// SPDX-License-Identifier: Apache-2.0
#include "../runtime/rindi_gdn_state.h"
#include <iostream>
#include <vector>

int main() {
    RindiGdnState state(1, 2, 2);
    const std::vector<uint16_t> q = {0x3c00, 0x3c00};
    const std::vector<uint16_t> k = {0x3c00, 0x3c00};
    const std::vector<uint16_t> v = {0x4000, 0x4000};
    const std::vector<uint16_t> decay = {0x3c00};
    const std::vector<uint16_t> beta = {0x3c00};
    std::vector<uint16_t> output;
    const auto initial = state.snapshot();
    state.step(q.data(), k.data(), v.data(), decay.data(), beta.data(), output);
    // First update: state[dv,:] = (2 - 0) * k = [2,2], q contraction is 4,
    // and the model's output scale is 64.
    bool ok = output.size() == 2 && output[0] == 0x5c00 && output[1] == 0x5c00;
    state.reset();
    ok = ok && state.restore(initial);
    std::vector<uint16_t> output_again;
    state.step(q.data(), k.data(), v.data(), decay.data(), beta.data(), output_again);
    ok = ok && output_again == output;
    std::cout << (ok ? "GDN_STATE=PASS\n" : "GDN_STATE=FAIL\n");
    return ok ? 0 : 1;
}
