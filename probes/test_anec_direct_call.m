#import <Foundation/Foundation.h>
#include <stdio.h>
#include <dlfcn.h>

int main() {
    @autoreleasepool {
        void* handle = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);
        if (!handle) { printf("No framework\n"); return 1; }
        
        void* sym_compile = dlsym(handle, "ANECCompile");
        void* sym_offline = dlsym(handle, "ANECCompileOffline");
        printf("ANECCompile: %p, ANECCompileOffline: %p\n", sym_compile, sym_offline);
        
        // Prepare a test directory in /tmp/test_anec_model
        NSString* dir = @"/tmp/test_anec_model";
        NSFileManager* fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:dir error:nil];
        [fm createDirectoryAtPath:[dir stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        
        size_t C = 64, S = 32, K = 4;
        NSMutableString* mil = [NSMutableString string];
        [mil appendString:@"program(1.3)\n"];
        [mil appendString:@"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"];
        [mil appendString:@"{\n"];
        [mil appendFormat:@"  func main<ios18>(tensor<fp16, [1, %zu, 1, %zu]> x) {\n", C, S];
        [mil appendFormat:@"    tensor<fp16, [%zu, 1, 1, %zu]> w = const()[name=string(\"w\"), val=tensor<fp16, [%zu, 1, 1, %zu]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n", C, K, C, K];
        [mil appendString:@"    tensor<int32, [2]> strides = const()[name=string(\"strides\"), val=tensor<int32, [2]>([1,1])];\n"];
        [mil appendString:@"    tensor<int32, [2]> dil = const()[name=string(\"dil\"), val=tensor<int32, [2]>([1,1])];\n"];
        [mil appendFormat:@"    tensor<int32, [4]> pad = const()[name=string(\"pad\"), val=tensor<int32, [4]>([0,0,%zu,0])];\n", K - 1];
        [mil appendFormat:@"    tensor<fp16, [1, %zu, 1, %zu]> c = conv(dilations=dil, groups=int32(%zu), pad=pad, pad_type=string(\"custom\"), strides=strides, weight=w, x=x)[name=string(\"c\")];\n", C, S, C];
        [mil appendFormat:@"    tensor<fp16, [1, %zu, 1, %zu]> y = mul(x=c, y=fp16(0x1p+0))[name=string(\"y\")];\n", C, S];
        [mil appendString:@"  } -> (y);\n"];
        [mil appendString:@"}\n"];
        [mil appendFormat:@"// qwen38_gdn_depthwise_C%zu\n", C];
        
        NSData* mil_data = [mil dataUsingEncoding:NSUTF8StringEncoding];
        [mil_data writeToFile:[dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        
        size_t weight_bytes = C * K * sizeof(uint16_t);
        NSMutableData* blob = [NSMutableData dataWithLength:128 + weight_bytes];
        uint32_t* words = (uint32_t*)[blob mutableBytes];
        words[0] = 1;
        words[1] = 2;
        words[16] = 0xDEADBEEF;
        words[17] = 1;
        words[18] = (uint32_t)weight_bytes;
        words[20] = 0x80;
        uint16_t* w_ptr = (uint16_t*)(((uint8_t*)[blob mutableBytes]) + 128);
        for (size_t i = 0; i < C * K; ++i) w_ptr[i] = 0x3c00;
        
        [blob writeToFile:[dir stringByAppendingPathComponent:@"weights/w.bin"] atomically:YES];
        
        printf("Created model in %s\n", [dir UTF8String]); fflush(stdout);
        
        // Try calling ANECCompile
        typedef int (*ANECCompileFn1)(const char* path);
        ANECCompileFn1 fn1 = (ANECCompileFn1)sym_compile;
        printf("Calling ANECCompile...\n"); fflush(stdout);
        int res1 = fn1([dir UTF8String]);
        printf("ANECCompile(dir) returned: %d\n", res1); fflush(stdout);
        
        NSArray* contents = [fm contentsOfDirectoryAtPath:dir error:nil];
        printf("Contents of dir after compile (%lu items):\n", (unsigned long)[contents count]); fflush(stdout);
        for (NSString* item in contents) {
            printf("  - %s\n", [item UTF8String]); fflush(stdout);
        }
    }
    return 0;
}
