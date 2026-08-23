// Compile an arbitrary MIL text file against a weights directory.
#include "runtime/ane_c_bridge.h"
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <dirent.h>
static std::vector<uint8_t> rf(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) return {};
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n); if (n && fread(v.data(), 1, n, f) != (size_t)n) v.clear();
    fclose(f); return v;
}
int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: mil weightsdir\n"); return 2; }
    auto mil = rf(argv[1]);
    if (mil.empty()) { printf("no mil\n"); return 2; }
    std::vector<std::string> names;
    std::vector<std::vector<uint8_t>> datas;
    DIR* d = opendir(argv[2]);
    if (!d) { printf("no weights dir\n"); return 2; }
    struct dirent* e;
    while ((e = readdir(d))) {
        std::string n = e->d_name;
        if (n.size() > 4 && n.substr(n.size()-4) == ".bin") {
            auto v = rf(std::string(argv[2]) + "/" + n);
            if (!v.empty()) { names.push_back(n); datas.push_back(std::move(v)); }
        }
    }
    closedir(d);
    std::vector<const char*> np; std::vector<const void*> dp; std::vector<size_t> sz;
    for (size_t i = 0; i < names.size(); ++i) {
        np.push_back(names[i].c_str()); dp.push_back(datas[i].data()); sz.push_back(datas[i].size());
    }
    ANEContext* ctx = ane_context_create();
    ANEModel* m = ane_model_compile_mil(ctx, std::string(mil.begin(), mil.end()).c_str(),
                                        np.data(), dp.data(), sz.data(), np.size(), 0, 21);
    printf("%s (%zu weight blobs)\n", m ? "COMPILED" : "REJECTED", np.size());
    if (m) ane_model_release(m);
    return m ? 0 : 1;
}
