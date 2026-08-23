#import <Foundation/Foundation.h>
#include <stdio.h>
#include <dlfcn.h>

typedef int (*ANECCompileFn)(const char* in_dir, const char* out_dir, NSDictionary* options);

int main() {
    @autoreleasepool {
        void* handle = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);
        if (!handle) {
            printf("Failed to load ANECompiler.framework: %s\n", dlerror());
            return 1;
        }
        printf("Loaded ANECompiler.framework successfully!\n");
        
        ANECCompileFn compile_fn = (ANECCompileFn)dlsym(handle, "ANECCompile");
        printf("ANECCompile symbol: %p\n", compile_fn);
    }
    return 0;
}
