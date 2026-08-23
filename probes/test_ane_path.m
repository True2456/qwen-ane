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
        NSString* mil = @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\nfunc main<ios18>(tensor<fp16, [1,4,1,32]> x) {\n  tensor<fp16, [1,4,1,32]> y = identity(x=x)[name=string(\"y\")];\n} -> (y);\n}\n";
        NSData* mil_data = [mil dataUsingEncoding:NSUTF8StringEncoding];
        
        typedef id (*DescFn)(id, SEL, NSData*, NSDictionary*, NSDictionary*);
        DescFn desc_fn = (DescFn)[desc_cls methodForSelector:desc_sel];
        id desc = desc_fn(desc_cls, desc_sel, mil_data, @{}, nil);
        
        Class model_cls = NSClassFromString(@"_ANEInMemoryModel");
        id model = ((id (*)(id, SEL, id))objc_msgSend)(model_cls, @selector(inMemoryModelWithDescriptor:), desc);
        
        id path_obj = ((id (*)(id, SEL))objc_msgSend)(model, @selector(localModelPath));
        if (path_obj) {
            printf("path_obj class: %s\n", [[path_obj className] UTF8String]);
            printf("path_obj is NSString: %d\n", [path_obj isKindOfClass:[NSString class]]);
            printf("path_obj is NSURL: %d\n", [path_obj isKindOfClass:[NSURL class]]);
            printf("path_obj description: %s\n", [[path_obj description] UTF8String]);
        } else {
            printf("path_obj is nil\n");
        }
    }
    return 0;
}
