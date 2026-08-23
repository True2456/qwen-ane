#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <stdio.h>
#include <dlfcn.h>

int main() {
    @autoreleasepool {
        void* handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        if (!handle) { printf("No framework\n"); return 1; }
        
        Class inmem_cls = NSClassFromString(@"_ANEInMemoryModel");
        NSString* path = @"/var/folders/08/kjxn753d64g6s12lt34lv_yr0000gn/T/D73F5B75B292B796DC9998852378F7EFE6C0FDF674F4F02A42BF572674CC88FD_7423974D4E18CB7791A67AC5A071B72542A0087342BF30856800481E0819C8EE_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";
        NSURL* url = [NSURL fileURLWithPath:path];
        
        id raw = [inmem_cls alloc];
        typedef id (*InitURLFn)(id, SEL, NSURL*, NSString*, NSInteger, NSDictionary*);
        InitURLFn init_fn = (InitURLFn)[raw methodForSelector:@selector(initWithURL:key:qos:options:)];
        id model = init_fn(raw, @selector(initWithURL:key:qos:options:), url, @"q38_layer", 21, @{});
        printf("initWithURL result: %p\n", model);
        
        if (model) {
            NSError* err = nil;
            typedef BOOL (*LoadFn)(id, SEL, NSInteger, NSDictionary*, NSError**);
            LoadFn load_fn = (LoadFn)[model methodForSelector:@selector(loadWithQoS:options:error:)];
            BOOL ok = load_fn(model, @selector(loadWithQoS:options:error:), 21, @{ @"kANEFProcedureVariantHint": @1 }, &err);
            printf("loadWithQoS result: %d\n", ok);
            if (err) printf("load error: %s\n", [[err description] UTF8String]);
        }
    }
    return 0;
}
