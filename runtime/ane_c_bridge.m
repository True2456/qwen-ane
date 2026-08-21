/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/ane_c_bridge.m - C bridge for Apple Silicon ANE Private Frameworks & Hardware Dispatch.
 */

#import "ane_c_bridge.h"
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>

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
};

struct ANERequest {
    id rawRequest;
    id inObj;
    id outObj;
    int procedureIndex;
};

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
    @autoreleasepool {
        NSString* pathStr = [NSString stringWithUTF8String:package_path];
        NSURL* fileURL = [NSURL fileURLWithPath:pathStr];
        NSString* keyStr = key ? [NSString stringWithUTF8String:key] : @"q38_layer";

        Class modelClass = ctx->aneInMemoryModelClass ? ctx->aneInMemoryModelClass : NSClassFromString(@"_ANEModel");
        if (!modelClass) return NULL;

        id rawModel = [modelClass alloc];
        SEL initSel = @selector(initWithURL:key:qos:options:);
        if ([rawModel respondsToSelector:initSel]) {
            typedef id (*InitFn)(id, SEL, NSURL*, NSString*, NSInteger, NSDictionary*);
            InitFn initMethod = (InitFn)[rawModel methodForSelector:initSel];
            id loaded = initMethod(rawModel, initSel, fileURL, keyStr, qos > 0 ? qos : 21, @{});
            if (loaded) {
                ANEModel* model = (ANEModel*)calloc(1, sizeof(ANEModel));
                model->rawModel = loaded;
                model->loadedModel = loaded;
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
                model->rawModel = loaded;
                model->loadedModel = loaded;
                return model;
            }
        }

        return NULL;
    }
}

void ane_model_release(ANEModel* model) {
    if (model) {
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
        SEL createObjSel = @selector(objectWithIOSurface:);
        if (![ctx->aneIOSurfaceObjectClass respondsToSelector:createObjSel]) return NULL;

        typedef id (*CreateObjFn)(id, SEL, IOSurfaceRef);
        CreateObjFn createObjMethod = (CreateObjFn)[ctx->aneIOSurfaceObjectClass methodForSelector:createObjSel];
        
        id inObj = createObjMethod(ctx->aneIOSurfaceObjectClass, createObjSel, input_surface);
        id outObj = createObjMethod(ctx->aneIOSurfaceObjectClass, createObjSel, output_surface);
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

        SEL evalSel = @selector(evaluateWithQoS:options:request:error:);
        if ([target respondsToSelector:evalSel]) {
            typedef BOOL (*EvalFn)(id, SEL, NSInteger, NSDictionary*, id, NSError**);
            EvalFn evalMethod = (EvalFn)[target methodForSelector:evalSel];
            NSError* err = nil;
            BOOL ok = evalMethod(target, evalSel, 21, @{}, req->rawRequest, &err);
            return ok && (err == nil);
        }

        SEL evalModelSel = @selector(evaluateWithModel:options:request:error:);
        if (ctx->aneClient && [ctx->aneClient respondsToSelector:evalModelSel]) {
            typedef BOOL (*EvalModelFn)(id, SEL, id, NSDictionary*, id, NSError**);
            EvalModelFn evalModelMethod = (EvalModelFn)[ctx->aneClient methodForSelector:evalModelSel];
            NSError* err = nil;
            id m = model->loadedModel ? model->loadedModel : model->rawModel;
            BOOL ok = evalModelMethod(ctx->aneClient, evalModelSel, m, @{}, req->rawRequest, &err);
            return ok && (err == nil);
        }

        return true;
    }
}
