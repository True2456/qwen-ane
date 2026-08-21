/*
 * SPDX-License-Identifier: Apache-2.0
 * ane_c_bridge.m - Implementation of pure C bridge to AppleNeuralEngine.framework.
 */

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include "ane_c_bridge.h"

struct ANEContext {
    id client;
    Class aneClientClass;
    Class aneModelClass;
    Class aneRequestClass;
    Class aneIOSurfaceObjectClass;
    void* dylibHandle;
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
        ANEContext* ctx = (ANEContext*)calloc(1, sizeof(ANEContext));
        if (!ctx) return NULL;

        ctx->dylibHandle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);

        ctx->aneClientClass = objc_getClass("_ANEClient");
        ctx->aneModelClass = objc_getClass("_ANEModel");
        ctx->aneRequestClass = objc_getClass("_ANERequest");
        ctx->aneIOSurfaceObjectClass = objc_getClass("_ANEIOSurfaceObject");

        if (!ctx->aneClientClass || !ctx->aneModelClass || !ctx->aneRequestClass || !ctx->aneIOSurfaceObjectClass) {
            free(ctx);
            return NULL;
        }

        // Connect to _ANEClient
        if ([ctx->aneClientClass respondsToSelector:@selector(sharedConnection)]) {
            ctx->client = [ctx->aneClientClass performSelector:@selector(sharedConnection)];
        } else {
            ctx->client = [[ctx->aneClientClass alloc] init];
        }

        return ctx;
    }
}

void ane_context_destroy(ANEContext* ctx) {
    if (ctx) {
        free(ctx);
    }
}

ANEModel* ane_model_load_compiled(ANEContext* ctx, const char* package_path, const char* key, int qos) {
    if (!ctx || !package_path) return NULL;
    @autoreleasepool {
        NSString* pathStr = [NSString stringWithUTF8String:package_path];
        NSURL* fileURL = [NSURL fileURLWithPath:pathStr];
        NSString* keyStr = [NSString stringWithUTF8String:(key ? key : "model_0")];

        id rawModel = [ctx->aneModelClass alloc];
        SEL initSel = @selector(initWithModelAtURL:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:);
        
        typedef id (*InitFn)(id, SEL, NSURL*, NSString*, NSInteger, NSString*, NSDictionary*, BOOL);
        InitFn initMethod = (InitFn)[rawModel methodForSelector:initSel];
        if (!initMethod) return NULL;

        id initializedModel = initMethod(rawModel, initSel, fileURL, keyStr, 1, keyStr, @{}, YES);
        if (!initializedModel) return NULL;

        ANEModel* model = (ANEModel*)calloc(1, sizeof(ANEModel));
        model->rawModel = initializedModel;
        model->loadedModel = initializedModel;
        return model;
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

        SEL reqSel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:);
        typedef id (*CreateReqFn)(id, SEL, NSArray*, NSArray*, NSArray*, NSArray*, id, NSNumber*);
        CreateReqFn createReqMethod = (CreateReqFn)[ctx->aneRequestClass methodForSelector:reqSel];

        id req = createReqMethod(ctx->aneRequestClass, reqSel, inArr, inIdx, outArr, outIdx, nil, procNum);
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
    ANERequest* req,
    void* wait_shared_event,
    uint64_t wait_value,
    void* signal_shared_event,
    uint64_t signal_value
) {
    if (!ctx || !req || !ctx->client) return false;
    @autoreleasepool {
        SEL evalSel = @selector(evaluateWithModel:options:request:qos:error:);
        typedef BOOL (*EvalFn)(id, SEL, id, NSDictionary*, id, NSInteger, NSError**);
        EvalFn evalMethod = (EvalFn)[ctx->client methodForSelector:evalSel];
        if (!evalMethod) return false;

        NSError* err = nil;
        BOOL ok = evalMethod(ctx->client, evalSel, req->rawRequest, @{}, req->rawRequest, 21, &err);
        return ok && (err == nil);
    }
}
