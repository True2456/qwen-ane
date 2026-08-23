#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <stdio.h>
#include <dlfcn.h>

int main() {
    @autoreleasepool {
        void* handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        if (!handle) { printf("No framework\n"); return 1; }
        
        Class desc_cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        SEL desc_sel = @selector(modelWithMILText:weights:optionsPlist:);
        
        size_t dim = 64;
        size_t width = 32;
        
        NSMutableString* mil = [NSMutableString string];
        [mil appendString:@"program(1.3)\n"];
        [mil appendString:@"[buildInfo = dict<string, string>({\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"})]\n"];
        [mil appendString:@"{\n"];
        [mil appendFormat:@"  func main<ios18>(tensor<fp16, [1, %zu, 1, %zu]> x) {\n", dim, width];
        [mil appendString:@"    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"];
        [mil appendString:@"    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"];
        [mil appendString:@"    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
        [mil appendString:@"    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"];
        [mil appendString:@"    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
        [mil appendFormat:@"    tensor<fp16, [%zu, %zu, 1, 1]> w = const()[name=string(\"w\"), val=tensor<fp16, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n", dim, dim, dim, dim];
        [mil appendFormat:@"    tensor<fp16, [1, %zu, 1, %zu]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"y\")];\n", dim, width];
        [mil appendString:@"  } -> (y);\n"];
        [mil appendString:@"}\n"];
        
        NSData* mil_data = [mil dataUsingEncoding:NSUTF8StringEncoding];
        
        size_t weight_bytes = dim * dim * sizeof(uint16_t);
        NSMutableData* raw_w = [NSMutableData dataWithLength:weight_bytes];
        uint16_t* ptr = (uint16_t*)[raw_w mutableBytes];
        for (size_t i = 0; i < dim * dim; ++i) ptr[i] = 0x3c00;
        
        // Wrap in DEADBEEF blob
        NSMutableData* blob = [NSMutableData dataWithLength:128 + weight_bytes];
        uint32_t* words = (uint32_t*)[blob mutableBytes];
        words[0] = 1;
        words[1] = 2;
        words[16] = 0xDEADBEEF;
        words[17] = 1;
        words[18] = (uint32_t)weight_bytes;
        words[20] = 0x80;
        memcpy(((uint8_t*)[blob mutableBytes]) + 128, [raw_w bytes], weight_bytes);
        
        NSDictionary* entry = @{ @"data": blob, @"offset": @0 };
        NSDictionary* weights = @{ @"@model_path/weights/w.bin": entry };
        
        typedef id (*DescFn)(id, SEL, NSData*, NSDictionary*, NSDictionary*);
        DescFn desc_fn = (DescFn)[desc_cls methodForSelector:desc_sel];
        id desc = desc_fn(desc_cls, desc_sel, mil_data, weights, nil);
        
        Class model_cls = NSClassFromString(@"_ANEInMemoryModel");
        id model = ((id (*)(id, SEL, id))objc_msgSend)(model_cls, @selector(inMemoryModelWithDescriptor:), desc);
        
        NSString* local = ((id (*)(id, SEL))objc_msgSend)(model, @selector(localModelPath));
        
        NSFileManager* fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:local error:nil];
        [fm createDirectoryAtPath:[local stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil_data writeToFile:[local stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [blob writeToFile:[local stringByAppendingPathComponent:@"weights/w.bin"] atomically:YES];
        
        NSDictionary* opts = @{ @"kANEFProcedureVariantHint": @1 };
        
        NSError* error = nil;
        typedef BOOL (*CompileFn)(id, SEL, NSInteger, NSDictionary*, NSError**);
        CompileFn comp = (CompileFn)[model methodForSelector:@selector(compileWithQoS:options:error:)];
        BOOL ok = comp(model, @selector(compileWithQoS:options:error:), 21, opts, &error);
        printf("compile result: %d\n", ok);
        if (error) {
            printf("compile error: %s\n", [[error description] UTF8String]);
        }
    }
    return 0;
}
