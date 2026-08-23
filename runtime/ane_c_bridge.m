/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/ane_c_bridge.m - C bridge for Apple Silicon ANE Private Frameworks & Hardware Dispatch.
 */

#import "ane_c_bridge.h"
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>

struct ANEContext {
    Class aneClientClass;
    Class aneInMemoryModelClass;
    Class aneRequestClass;
    Class aneIOSurfaceObjectClass;
    id aneClient;
};

struct ANEModel {
    id rawModel;
    id loadedModel;
    size_t outputChannels[8];
    size_t outputCount;
};

static size_t ane_parse_output_channels(id model, size_t* channels, size_t capacity) {
    if (!model || !channels || capacity == 0) return 0;
    id inner = model;
    SEL modelSel = @selector(model);
    if ([inner respondsToSelector:modelSel]) {
        id candidate = ((id (*)(id, SEL))objc_msgSend)(inner, modelSel);
        if (candidate) inner = candidate;
    }
    if (![inner respondsToSelector:@selector(description)]) return 0;
    NSString* descObj = ((id (*)(id, SEL))objc_msgSend)(inner, @selector(description));
    const char* desc = [descObj isKindOfClass:[NSString class]]
        ? [(NSString*)descObj UTF8String] : NULL;
    if (!desc) return 0;

    // The private ANE description identifies output symbols as a channel
    // dimension followed by Name = "...@output".  Keep this parser small and
    // independent of the private protobuf classes; it mirrors the working
    // Python driver, which also binds by channel dimension.
    size_t count = 0;
    const char* p = desc;
    while (*p && count < capacity) {
        const char* ch = strstr(p, "Channels = ");
        if (!ch) break;
        unsigned long value = strtoul(ch + 11, NULL, 10);
        const char* next = strstr(ch + 11, "Channels = ");
        const char* name = strstr(ch, "Name = ");
        if (name && (!next || name < next)) {
            const char* at = strstr(name, "@output");
            if (at && (!next || at < next)) channels[count++] = (size_t)value;
        }
        p = ch + 11;
    }
    return count;
}

static NSData* ane_make_blob(const void* data, size_t size) {
    NSMutableData* blob = [NSMutableData dataWithLength:128 + size];
    uint8_t* bytes = (uint8_t*)[blob mutableBytes];
    uint32_t* words = (uint32_t*)bytes;
    words[0] = 1;
    words[1] = 2;
    words[16] = 0xDEADBEEF;
    words[17] = 1;
    words[18] = (uint32_t)size;
    words[20] = 0x80;
    if (size > 0 && data) memcpy(bytes + 128, data, size);
    return blob;
}

static NSDictionary* ane_compile_options(int instance_hint) {
    NSMutableDictionary* options = [NSMutableDictionary dictionary];
    [options setObject:[NSNumber numberWithInt:1]
                forKey:@"kANEFProcedureVariantHint"];
    if (instance_hint > 0) {
        [options setObject:[NSNumber numberWithInt:instance_hint]
                    forKey:@"kANEFAneInstanceHint"];
    }
    const char* keep_wired = getenv("Q38_ANE_KEEP_WIRED");
    if (keep_wired) {
        [options setObject:[NSNumber numberWithInt:atoi(keep_wired)]
                    forKey:@"kANEFKeepModelMemoryWiredKey"];
    }
    return options;
}

static BOOL ane_call_bool(id object, SEL selector, NSInteger qos,
                          NSDictionary* options, NSError** error) {
    if (!object || ![object respondsToSelector:selector]) return NO;
    typedef BOOL (*Fn)(id, SEL, NSInteger, NSDictionary*, NSError**);
    Fn fn = (Fn)[object methodForSelector:selector];
    return fn(object, selector, qos, options, error);
}

static NSString* ane_model_path(id model) {
    SEL selector = @selector(localModelPath);
    if (!model || ![model respondsToSelector:selector]) return nil;
    id value = ((id (*)(id, SEL))objc_msgSend)(model, selector);
    return [value isKindOfClass:[NSString class]] ? (NSString*)value : nil;
}

ANEModel* ane_model_compile_mil(
    ANEContext* ctx,
    const char* mil_text,
    const char* const* weight_names,
    const void* const* weight_data,
    const size_t* weight_sizes,
    size_t weight_count,
    int instance_hint,
    int qos
) {
    if (!ctx || !mil_text || !ctx->aneInMemoryModelClass) return NULL;

    @autoreleasepool {
        NSMutableDictionary* weights = [NSMutableDictionary dictionary];
        for (size_t i = 0; i < weight_count; ++i) {
            if (!weight_names[i] || (!weight_data[i] && weight_sizes[i] != 0)) return NULL;
            NSString* path = [NSString stringWithFormat:@"@model_path/weights/%s",
                              weight_names[i]];
            NSData* blob = ane_make_blob(weight_data[i], weight_sizes[i]);
            NSDictionary* entry = @{ @"data": blob, @"offset": @0 };
            [weights setObject:entry forKey:path];
        }

        Class descriptor_class = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        SEL descriptor_selector = @selector(modelWithMILText:weights:optionsPlist:);
        if (!descriptor_class || ![descriptor_class respondsToSelector:descriptor_selector]) {
            NSLog(@"[ANE Bridge] MIL descriptor API unavailable");
            return NULL;
        }
        NSData* mil_data = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        typedef id (*DescriptorFn)(id, SEL, NSData*, NSDictionary*, NSDictionary*);
        DescriptorFn descriptor_fn = (DescriptorFn)[descriptor_class methodForSelector:descriptor_selector];
        id descriptor = descriptor_fn(descriptor_class, descriptor_selector, mil_data, weights, nil);
        if (!descriptor) {
            NSLog(@"[ANE Bridge] Failed to create MIL descriptor");
            return NULL;
        }

        SEL model_selector = @selector(inMemoryModelWithDescriptor:);
        if (![ctx->aneInMemoryModelClass respondsToSelector:model_selector]) return NULL;
        id model = ((id (*)(id, SEL, id))objc_msgSend)(ctx->aneInMemoryModelClass,
                                                        model_selector, descriptor);
        if (!model) return NULL;

        NSDictionary* options = ane_compile_options(instance_hint);
        BOOL loaded = NO;
        SEL exists_selector = @selector(compiledModelExists);
        if ([model respondsToSelector:exists_selector]) {
            loaded = ((BOOL (*)(id, SEL))objc_msgSend)(model, exists_selector);
            if (loaded) {
                NSError* error = nil;
                loaded = ane_call_bool(model, @selector(loadWithQoS:options:error:),
                                       qos > 0 ? qos : 21, options, &error);
                if (!loaded && error) NSLog(@"[ANE Bridge] Cached load failed: %@", error);
            }
        }

        if (!loaded) {
            NSString* local = ane_model_path(model);
            if (!local) {
                NSLog(@"[ANE Bridge] Model has no localModelPath");
                return NULL;
            }
            NSFileManager* fm = [NSFileManager defaultManager];
            [fm removeItemAtPath:local error:nil];
            [fm createDirectoryAtPath:[local stringByAppendingPathComponent:@"weights"]
           withIntermediateDirectories:YES attributes:nil error:nil];
            NSString* mil_path = [local stringByAppendingPathComponent:@"model.mil"];
            [mil_data writeToFile:mil_path atomically:YES];
            for (size_t i = 0; i < weight_count; ++i) {
                NSString* path = [[local stringByAppendingPathComponent:@"weights"]
                                  stringByAppendingPathComponent:
                                  [NSString stringWithUTF8String:weight_names[i]]];
                NSData* blob = [weights objectForKey:
                    [NSString stringWithFormat:@"@model_path/weights/%s", weight_names[i]]][@"data"];
                [blob writeToFile:path atomically:YES];
            }
            NSError* error = nil;
            BOOL compiled = ane_call_bool(model, @selector(compileWithQoS:options:error:),
                                          qos > 0 ? qos : 21, options, &error);
            if (!compiled) {
                NSLog(@"[ANE Bridge] MIL compile failed: %@", error);
                return NULL;
            }
            error = nil;
            loaded = ane_call_bool(model, @selector(loadWithQoS:options:error:),
                                   qos > 0 ? qos : 21, options, &error);
            if (!loaded) {
                NSLog(@"[ANE Bridge] MIL load failed: %@", error);
                return NULL;
            }
        }

        ANEModel* result = (ANEModel*)calloc(1, sizeof(ANEModel));
        result->rawModel = [model retain];
        result->loadedModel = result->rawModel;
        result->outputCount = ane_parse_output_channels(result->loadedModel,
                                                        result->outputChannels,
                                                        8);
        return result;
    }
}

size_t ane_model_output_channels(ANEModel* model, size_t* channels, size_t capacity) {
    if (!model || !channels || capacity == 0) return 0;
    size_t count = model->outputCount < capacity ? model->outputCount : capacity;
    if (count == 0) {
        count = ane_parse_output_channels(model->loadedModel ? model->loadedModel
                                                              : model->rawModel,
                                          channels, capacity);
    } else {
        memcpy(channels, model->outputChannels, count * sizeof(size_t));
    }
    return count;
}

struct ANERequest {
    id rawRequest;
    id inObj;
    id outObj;
    int procedureIndex;
};

static id ane_wrap_iosurface(ANEContext* ctx, IOSurfaceRef surface) {
    if (!ctx || !surface || !ctx->aneIOSurfaceObjectClass) return nil;
    SEL allocSel = @selector(alloc);
    SEL initSel = @selector(initWithIOSurface:startOffset:shouldRetain:);
    if (![ctx->aneIOSurfaceObjectClass respondsToSelector:allocSel] ||
        ![ctx->aneIOSurfaceObjectClass instancesRespondToSelector:initSel]) return nil;
    id object = ((id (*)(id, SEL))objc_msgSend)(ctx->aneIOSurfaceObjectClass, allocSel);
    typedef id (*InitFn)(id, SEL, IOSurfaceRef, id, BOOL);
    InitFn init = (InitFn)[object methodForSelector:initSel];
    return init(object, initSel, surface, [NSNumber numberWithUnsignedLong:0], YES);
}

ANEContext* ane_context_create(void) {
    @autoreleasepool {
        void* handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        if (!handle) {
            NSLog(@"[ANE Bridge] Failed to load AppleNeuralEngine.framework");
            return NULL;
        }

        ANEContext* ctx = (ANEContext*)calloc(1, sizeof(ANEContext));
        ctx->aneClientClass = NSClassFromString(@"_ANEClient");
        ctx->aneInMemoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
        ctx->aneRequestClass = NSClassFromString(@"_ANERequest");
        ctx->aneIOSurfaceObjectClass = NSClassFromString(@"_ANEIOSurfaceObject");

        if (ctx->aneClientClass) {
            SEL sharedConnSel = @selector(sharedConnection);
            if ([ctx->aneClientClass respondsToSelector:sharedConnSel]) {
                ctx->aneClient = ((id (*)(id, SEL))objc_msgSend)(ctx->aneClientClass, sharedConnSel);
            }
        }
        return ctx;
    }
}

void ane_context_destroy(ANEContext* ctx) {
    if (ctx) {
        free(ctx);
    }
}

ANEModel* ane_model_load_compiled(
    ANEContext* ctx,
    const char* package_path,
    const char* key,
    int qos
) {
    if (!ctx || !package_path) return NULL;
    if (access(package_path, F_OK) != 0) return NULL;

    @autoreleasepool {
        NSString* pathStr = [NSString stringWithUTF8String:package_path];
        NSURL* fileURL = [NSURL fileURLWithPath:pathStr];
        NSString* keyStr = key ? [NSString stringWithUTF8String:key] : @"q38_layer";

        Class modelClass = NSClassFromString(@"_ANEModel");
        if (!modelClass) return NULL;

        id rawModel = [modelClass alloc];
        // The exported ANE cache directories contain the native `__.bin` and
        // `__s.bin` model package.  They are not MIL source directories, so
        // `_ANEInMemoryModel initWithURL:key:qos:options:` cannot reopen them.
        // Use the same direct package initializer as the working Python
        // driver, then register the package with _ANEClient.
        SEL directInitSel = @selector(initWithModelAtURL:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:);
        if ([rawModel respondsToSelector:directInitSel]) {
            typedef id (*DirectInitFn)(id, SEL, NSURL*, NSString*, NSInteger,
                                       NSString*, NSDictionary*, BOOL);
            DirectInitFn initMethod = (DirectInitFn)[rawModel methodForSelector:directInitSel];
            id loaded = initMethod(rawModel, directInitSel, fileURL,
                                   keyStr, 1, keyStr, @{}, YES);
            if (loaded && ctx->aneClient &&
                [ctx->aneClient respondsToSelector:@selector(loadModel:options:qos:error:)]) {
                NSDictionary* opts = ane_compile_options(0);
                NSError* error = nil;
                typedef BOOL (*LoadFn)(id, SEL, id, NSDictionary*, unsigned int, NSError**);
                LoadFn loadMethod = (LoadFn)[ctx->aneClient methodForSelector:@selector(loadModel:options:qos:error:)];
                BOOL ok = loadMethod(ctx->aneClient, @selector(loadModel:options:qos:error:),
                                     loaded, opts, qos > 0 ? qos : 21, &error);
                if (ok) {
                    ANEModel* model = (ANEModel*)calloc(1, sizeof(ANEModel));
                    model->rawModel = [loaded retain];
                    model->loadedModel = model->rawModel;
                    return model;
                }
                if (error) NSLog(@"[ANE Bridge] Direct package load failed: %@", error);
            }
        }

        // Older systems expose only the in-memory URL initializer. Keep it as
        // a fallback for source/model directories produced by that API.
        modelClass = ctx->aneInMemoryModelClass ? ctx->aneInMemoryModelClass : NSClassFromString(@"_ANEModel");
        if (!modelClass) return NULL;
        rawModel = [modelClass alloc];
        SEL initSel = @selector(initWithURL:key:qos:options:);
        if ([rawModel respondsToSelector:initSel]) {
            typedef id (*InitFn)(id, SEL, NSURL*, NSString*, NSInteger, NSDictionary*);
            InitFn initMethod = (InitFn)[rawModel methodForSelector:initSel];
            id loaded = initMethod(rawModel, initSel, fileURL, keyStr, qos > 0 ? qos : 21, @{});
            if (loaded) {
                ANEModel* model = (ANEModel*)calloc(1, sizeof(ANEModel));
                model->rawModel = [loaded retain];
                model->loadedModel = model->rawModel;
                return model;
            }
        }

        SEL altInitSel = @selector(initWithURL:key:modelType:name:options:cache:);
        if ([rawModel respondsToSelector:altInitSel]) {
            typedef id (*AltInitFn)(id, SEL, NSURL*, NSString*, NSInteger, NSString*, NSDictionary*, BOOL);
            AltInitFn altInitMethod = (AltInitFn)[rawModel methodForSelector:altInitSel];
            id loaded = altInitMethod(rawModel, altInitSel, fileURL, keyStr, 1, keyStr, @{}, YES);
            if (loaded) {
                ANEModel* model = (ANEModel*)calloc(1, sizeof(ANEModel));
                model->rawModel = [loaded retain];
                model->loadedModel = model->rawModel;
                return model;
            }
        }

        return NULL;
    }
}

void ane_model_release(ANEModel* model) {
    if (model) {
        if (model->loadedModel) [model->loadedModel release];
        else if (model->rawModel) [model->rawModel release];
        free(model);
    }
}

ANERequest* ane_request_create(
    ANEContext* ctx,
    ANEModel* model,
    IOSurfaceRef input_surface,
    IOSurfaceRef output_surface,
    int procedure_index
) {
    if (!ctx || !model || !input_surface || !output_surface) return NULL;
    @autoreleasepool {
        id inObj = ane_wrap_iosurface(ctx, input_surface);
        id outObj = ane_wrap_iosurface(ctx, output_surface);
        if (!inObj || !outObj) return NULL;

        NSArray* inArr = @[inObj];
        NSArray* inIdx = @[@0];
        NSArray* outArr = @[outObj];
        NSArray* outIdx = @[@0];
        NSNumber* procNum = [NSNumber numberWithInt:procedure_index];

        id rawReq = [ctx->aneRequestClass alloc];
        SEL initReqSel = @selector(initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:);
        if (![rawReq respondsToSelector:initReqSel]) return NULL;

        typedef id (*InitReqFn)(id, SEL, NSArray*, NSArray*, NSArray*, NSArray*, id, id, NSNumber*, id, id);
        InitReqFn initReqMethod = (InitReqFn)[rawReq methodForSelector:initReqSel];
        
        id req = initReqMethod(rawReq, initReqSel, inArr, inIdx, outArr, outIdx, nil, nil, procNum, nil, nil);
        if (!req) return NULL;

        ANERequest* r = (ANERequest*)calloc(1, sizeof(ANERequest));
        r->rawRequest = req;
        r->inObj = inObj;
        r->outObj = outObj;
        r->procedureIndex = procedure_index;
        return r;
    }
}

ANERequest* ane_request_create_2in(
    ANEContext* ctx,
    ANEModel* model,
    IOSurfaceRef input1,
    IOSurfaceRef input2,
    IOSurfaceRef output_surface,
    int procedure_index
) {
    if (!ctx || !model || !input1 || !input2 || !output_surface) return NULL;
    @autoreleasepool {
        id in1Obj = ane_wrap_iosurface(ctx, input1);
        id in2Obj = ane_wrap_iosurface(ctx, input2);
        id outObj = ane_wrap_iosurface(ctx, output_surface);
        if (!in1Obj || !in2Obj || !outObj) return NULL;
        // EMPIRICAL ISA FINDING (ane-as test [6]): _ANERequest input array
        // positions map to MIL function parameters in REVERSE order - with
        // @[a, b] the hardware computed conv(param1<-b) and conv(param2<-a).
        // Reverse here so callers may pass surfaces in MIL parameter order.
        NSArray* inputs = @[in2Obj, in1Obj];
        NSArray* inputIndices = @[@0, @1];
        NSArray* outputs = @[outObj];
        NSArray* outputIndices = @[@0];
        id rawReq = [ctx->aneRequestClass alloc];
        SEL initSel = @selector(initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:);
        if (![rawReq respondsToSelector:initSel]) return NULL;
        typedef id (*InitFn)(id, SEL, NSArray*, NSArray*, NSArray*, NSArray*, id, id, NSNumber*, id, id);
        InitFn init = (InitFn)[rawReq methodForSelector:initSel];
        id request = init(rawReq, initSel, inputs, inputIndices, outputs, outputIndices,
                          nil, nil, [NSNumber numberWithInt:procedure_index], nil, nil);
        if (!request) return NULL;
        ANERequest* r = (ANERequest*)calloc(1, sizeof(ANERequest));
        r->rawRequest = request;
        r->inObj = in1Obj;
        r->outObj = outObj;
        r->procedureIndex = procedure_index;
        return r;
    }
}

bool ane_request_evaluate_realtime(
    ANEContext* ctx,
    ANEModel* model,
    ANERequest* req
) {
    if (!ctx || !model || !req || !req->rawRequest) return false;
    if (!ctx->aneClient ||
        ![ctx->aneClient respondsToSelector:@selector(evaluateRealTimeWithModel:options:request:error:)])
        return false;
    @autoreleasepool {
        id target = model->loadedModel ? model->loadedModel : ctx->aneClient;
        SEL sel = @selector(evaluateRealTimeWithModel:options:request:error:);
        typedef BOOL (*RtFn)(id, SEL, id, NSDictionary*, id, NSError**);
        RtFn fn = (RtFn)[ctx->aneClient methodForSelector:sel];
        NSError* err = nil;
        return fn(ctx->aneClient, sel, target, @{}, req->rawRequest, &err);
    }
}

ANERequest* ane_request_create_multi(
    ANEContext* ctx,
    ANEModel* model,
    IOSurfaceRef input_surface,
    IOSurfaceRef* output_surfaces,
    size_t output_count,
    int procedure_index
) {
    if (!ctx || !model || !input_surface || !output_surfaces || output_count == 0) return NULL;
    @autoreleasepool {
        id inObj = ane_wrap_iosurface(ctx, input_surface);
        if (!inObj) return NULL;
        NSMutableArray* outputs = [NSMutableArray arrayWithCapacity:output_count];
        NSMutableArray* outputIndices = [NSMutableArray arrayWithCapacity:output_count];
        for (size_t i = 0; i < output_count; ++i) {
            if (!output_surfaces[i]) return NULL;
            id outObj = ane_wrap_iosurface(ctx, output_surfaces[i]);
            if (!outObj) return NULL;
            [outputs addObject:outObj];
            [outputIndices addObject:[NSNumber numberWithUnsignedLong:i]];
        }
        NSArray* inputs = @[inObj];
        NSArray* inputIndices = @[@0];
        id rawReq = [ctx->aneRequestClass alloc];
        SEL initSel = @selector(initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:);
        if (![rawReq respondsToSelector:initSel]) return NULL;
        typedef id (*InitFn)(id, SEL, NSArray*, NSArray*, NSArray*, NSArray*, id, id, NSNumber*, id, id);
        InitFn init = (InitFn)[rawReq methodForSelector:initSel];
        id request = init(rawReq, initSel, inputs, inputIndices, outputs, outputIndices,
                          nil, nil, [NSNumber numberWithInt:procedure_index], nil, nil);
        if (!request) return NULL;
        ANERequest* result = (ANERequest*)calloc(1, sizeof(ANERequest));
        result->rawRequest = request;
        result->inObj = inObj;
        return result;
    }
}

void ane_request_release(ANERequest* req) {
    if (req) {
        free(req);
    }
}

bool ane_request_evaluate(
    ANEContext* ctx,
    ANEModel* model,
    ANERequest* req,
    void* wait_shared_event,
    uint64_t wait_value,
    void* signal_shared_event,
    uint64_t signal_value
) {
    if (!ctx || !model || !req || !req->rawRequest) return false;
    @autoreleasepool {
        id target = model->loadedModel ? model->loadedModel : ctx->aneClient;
        if (!target) return false;

        // Match the working Python driver: evaluate the loaded in-memory
        // model directly.  The client-level direct-evaluate selectors can
        // return before the output IOSurface has been committed, which is
        // especially visible for causal convolution windows reused at decode.
        SEL modelEvalSel = @selector(evaluateWithQoS:options:request:error:);
        if ([target respondsToSelector:modelEvalSel]) {
            typedef BOOL (*EvalFn)(id, SEL, NSInteger, NSDictionary*, id, NSError**);
            EvalFn evalMethod = (EvalFn)[target methodForSelector:modelEvalSel];
            NSError* err = nil;
            BOOL ok = evalMethod(target, modelEvalSel, 21, @{}, req->rawRequest, &err);
            if (!ok) NSLog(@"[ANE Bridge] In-memory evaluate returned false: %@", err ? [err description] : @"no error");
            if (ok && !err) return YES;   // success uses in-memory path
            // else fall through to direct-client failover below (needed for
            // width>32 programs that compile but don't evaluate in-memory).
        }

        // Directly loaded cache packages are owned by _ANEClient rather than
        // _ANEInMemoryModel.  Prefer the direct client entry points.
        if (ctx->aneClient && [ctx->aneClient respondsToSelector:@selector(doEvaluateDirectWithModel:options:request:qos:error:)]) {
            typedef BOOL (*EvalDirectFn)(id, SEL, id, NSDictionary*, id, unsigned int, NSError**);
            EvalDirectFn evalDirect = (EvalDirectFn)[ctx->aneClient methodForSelector:@selector(doEvaluateDirectWithModel:options:request:qos:error:)];
            NSError* err = nil;
            BOOL ok = evalDirect(ctx->aneClient, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                                 model->loadedModel ? model->loadedModel : model->rawModel,
                                 @{}, req->rawRequest, 21, &err);
            if (!ok) NSLog(@"[ANE Bridge] Direct evaluate returned false: %@", err ? [err description] : @"no error");
            return ok && (err == nil);
        }

        if (ctx->aneClient && [ctx->aneClient respondsToSelector:@selector(evaluateWithModel:options:request:qos:error:)]) {
            typedef BOOL (*EvalModelQosFn)(id, SEL, id, NSDictionary*, id, unsigned int, NSError**);
            EvalModelQosFn evalModel = (EvalModelQosFn)[ctx->aneClient methodForSelector:@selector(evaluateWithModel:options:request:qos:error:)];
            NSError* err = nil;
            BOOL ok = evalModel(ctx->aneClient, @selector(evaluateWithModel:options:request:qos:error:),
                                model->loadedModel ? model->loadedModel : model->rawModel,
                                @{}, req->rawRequest, 21, &err);
            if (!ok) NSLog(@"[ANE Bridge] Model evaluate returned false: %@", err ? [err description] : @"no error");
            return ok && (err == nil);
        }

        SEL evalSel = @selector(evaluateWithQoS:options:request:error:);
        if ([target respondsToSelector:evalSel]) {
            typedef BOOL (*EvalFn)(id, SEL, NSInteger, NSDictionary*, id, NSError**);
            EvalFn evalMethod = (EvalFn)[target methodForSelector:evalSel];
            NSError* err = nil;
            BOOL ok = evalMethod(target, evalSel, 21, @{}, req->rawRequest, &err);
            if (!ok) NSLog(@"[ANE Bridge] In-memory evaluate returned false: %@", err ? [err description] : @"no error");
            return ok && (err == nil);
        }

        SEL evalModelSel = @selector(evaluateWithModel:options:request:error:);
        if (ctx->aneClient && [ctx->aneClient respondsToSelector:evalModelSel]) {
            typedef BOOL (*EvalModelFn)(id, SEL, id, NSDictionary*, id, NSError**);
            EvalModelFn evalModelMethod = (EvalModelFn)[ctx->aneClient methodForSelector:evalModelSel];
            NSError* err = nil;
            id m = model->loadedModel ? model->loadedModel : model->rawModel;
            BOOL ok = evalModelMethod(ctx->aneClient, evalModelSel, m, @{}, req->rawRequest, &err);
            if (!ok) NSLog(@"[ANE Bridge] Legacy model evaluate returned false: %@", err ? [err description] : @"no error");
            return ok && (err == nil);
        }

        NSLog(@"[ANE Bridge] No evaluation selector available on loaded model/client");
        return false;
    }
}
