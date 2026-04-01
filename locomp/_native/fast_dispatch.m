// locust/_native/fast_dispatch.m
// Minimal C bridge for Metal kernel dispatch.
// Replaces 8+ PyObjC round-trips with a single ctypes call.

#import <Metal/Metal.h>

// Pending command buffer for async dispatch
static id<MTLCommandBuffer> _pending_cmd_buf = nil;

// Dispatch a compute kernel synchronously. Returns GPU time in milliseconds.
double locust_dispatch(void *queue_ptr, void *pipeline_ptr,
                       void **buffer_ptrs, int num_buffers,
                       int gx, int gy, int gz,
                       int tx, int ty, int tz) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_ptr;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        for (int i = 0; i < num_buffers; i++) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptrs[i];
            [encoder setBuffer:buf offset:0 atIndex:i];
        }

        MTLSize grid = MTLSizeMake(gx, gy, gz);
        MTLSize tgSize = MTLSizeMake(tx, ty, tz);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tgSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        return (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
    }
}

// Dispatch async — commit but don't wait. Returns immediately.
void locust_dispatch_async(void *queue_ptr, void *pipeline_ptr,
                           void **buffer_ptrs, int num_buffers,
                           int gx, int gy, int gz,
                           int tx, int ty, int tz) {
    // Wait for any previous pending work first
    if (_pending_cmd_buf != nil) {
        [_pending_cmd_buf waitUntilCompleted];
        _pending_cmd_buf = nil;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_ptr;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    for (int i = 0; i < num_buffers; i++) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptrs[i];
        [encoder setBuffer:buf offset:0 atIndex:i];
    }

    MTLSize grid = MTLSizeMake(gx, gy, gz);
    MTLSize tgSize = MTLSizeMake(tx, ty, tz);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tgSize];
    [encoder endEncoding];

    [cmdBuf commit];
    _pending_cmd_buf = cmdBuf;
}

// Wait for pending async work. Returns 0 if nothing pending.
double locust_sync(void) {
    if (_pending_cmd_buf == nil) return 0.0;
    [_pending_cmd_buf waitUntilCompleted];
    double gpu_ms = (_pending_cmd_buf.GPUEndTime - _pending_cmd_buf.GPUStartTime) * 1000.0;
    _pending_cmd_buf = nil;
    return gpu_ms;
}

// Dispatch same kernel N times in one command buffer. Returns avg GPU time in ms.
double locust_dispatch_repeat(void *queue_ptr, void *pipeline_ptr,
                              void **buffer_ptrs, int num_buffers,
                              int gx, int gy, int gz,
                              int tx, int ty, int tz,
                              int repeat) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_ptr;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

        MTLSize grid = MTLSizeMake(gx, gy, gz);
        MTLSize tgSize = MTLSizeMake(tx, ty, tz);

        for (int r = 0; r < repeat; r++) {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            for (int i = 0; i < num_buffers; i++) {
                id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptrs[i];
                [encoder setBuffer:buf offset:0 atIndex:i];
            }
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tgSize];
            [encoder endEncoding];
        }

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        double total_ms = (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
        return total_ms / repeat;
    }
}

// --- Batch mode ---
// Multiple different kernels in one command buffer.
static id<MTLCommandBuffer> _batch_cmd_buf = nil;

void locust_batch_begin(void *queue_ptr) {
    if (_batch_cmd_buf != nil) {
        [_batch_cmd_buf waitUntilCompleted];
        _batch_cmd_buf = nil;
    }
    // Also drain any pending async work
    if (_pending_cmd_buf != nil) {
        [_pending_cmd_buf waitUntilCompleted];
        _pending_cmd_buf = nil;
    }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    _batch_cmd_buf = [queue commandBuffer];
}

void locust_batch_dispatch(void *pipeline_ptr,
                           void **buffer_ptrs, int num_buffers,
                           int gx, int gy, int gz,
                           int tx, int ty, int tz) {
    if (_batch_cmd_buf == nil) return;

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_ptr;

    id<MTLComputeCommandEncoder> encoder = [_batch_cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    for (int i = 0; i < num_buffers; i++) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptrs[i];
        [encoder setBuffer:buf offset:0 atIndex:i];
    }

    MTLSize grid = MTLSizeMake(gx, gy, gz);
    MTLSize tgSize = MTLSizeMake(tx, ty, tz);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
}

double locust_batch_end(void) {
    if (_batch_cmd_buf == nil) return 0.0;
    [_batch_cmd_buf commit];
    [_batch_cmd_buf waitUntilCompleted];
    double gpu_ms = (_batch_cmd_buf.GPUEndTime - _batch_cmd_buf.GPUStartTime) * 1000.0;
    _batch_cmd_buf = nil;
    return gpu_ms;
}
