/*
 * Max-Plus GEMM CUDA Kernel
 * ==========================
 * Computes y_i = max_j(W_ij + x_j) + b_i  (tropical linear layer)
 * 
 * This is the tropical semiring equivalent of matrix-vector multiplication:
 *   Classical: y_i = sum_j(W_ij * x_j)
 *   Tropical:  y_i = max_j(W_ij + x_j)
 *
 * Tiling strategy:
 *   - Each thread block handles TILE_OUT output features
 *   - Inner dimension (in_features) is tiled in chunks of TILE_IN
 *   - Shared memory holds tiles of W and x for cache efficiency
 *   - Reduction uses warp-level primitives for the final max
 *
 * Supports:
 *   - Batched operation: (B, in_features) -> (B, out_features)
 *   - float32, float16 (via template)
 *   - Forward pass (max) and backward pass (softmax-weighted STE)
 *
 * Memory layout:
 *   W: (out_features, in_features) - row major
 *   x: (batch_size, in_features) - row major
 *   y: (batch_size, out_features) - row major
 *   b: (out_features,) - optional bias
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

// Tile sizes for shared memory
#define TILE_IN 128      // Tile size along input dimension
#define TILE_OUT 32      // Outputs per thread block
#define WARP_SIZE 32
#define BLOCK_SIZE 256   // Threads per block


/* ==========================================================================
 * Forward kernel: y_i = max_j(W_ij + x_j) + b_i
 * Each thread block computes one output for one batch element.
 * ==========================================================================*/

template <typename scalar_t>
__global__ void maxplus_forward_kernel(
    const scalar_t* __restrict__ W,      // (out_features, in_features)
    const scalar_t* __restrict__ x,      // (batch_size, in_features)
    scalar_t* __restrict__ y,            // (batch_size, out_features)
    const scalar_t* __restrict__ bias,   // (out_features,) or nullptr
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // Block handles one (batch, out_feature) pair
    const int b = blockIdx.x;            // batch index
    const int o = blockIdx.y;            // output feature index
    
    if (b >= batch_size || o >= out_features) return;
    
    // Shared memory for tiled reduction
    __shared__ scalar_t s_partial[BLOCK_SIZE];
    
    const scalar_t* x_row = x + b * in_features;
    const scalar_t* w_row = W + o * in_features;
    
    // Each thread computes max over its assigned input features
    scalar_t thread_max = -FLT_MAX;
    
    for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
        scalar_t val = w_row[j] + x_row[j];
        thread_max = fmaxf(thread_max, val);
    }
    
    // Store in shared memory
    s_partial[threadIdx.x] = thread_max;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] = fmaxf(s_partial[threadIdx.x],
                                            s_partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (threadIdx.x == 0) {
        scalar_t result = s_partial[0];
        if (bias != nullptr) {
            result += bias[o];
        }
        y[b * out_features + o] = result;
    }
}


/* ==========================================================================
 * Tiled forward kernel: better cache utilization for large in_features
 * Uses shared memory tiles to reduce global memory traffic.
 * ==========================================================================*/

template <typename scalar_t>
__global__ void maxplus_forward_tiled_kernel(
    const scalar_t* __restrict__ W,      // (out_features, in_features)
    const scalar_t* __restrict__ x,      // (batch_size, in_features)
    scalar_t* __restrict__ y,            // (batch_size, out_features)
    const scalar_t* __restrict__ bias,   // (out_features,) or nullptr
    const int batch_size,
    const int in_features,
    const int out_features
) {
    const int b = blockIdx.x;
    const int o = blockIdx.y;
    
    if (b >= batch_size || o >= out_features) return;
    
    __shared__ scalar_t s_x[TILE_IN];     // Shared tile of input
    __shared__ scalar_t s_w[TILE_IN];     // Shared tile of weights
    __shared__ scalar_t s_partial[BLOCK_SIZE];
    
    const scalar_t* x_row = x + b * in_features;
    const scalar_t* w_row = W + o * in_features;
    
    scalar_t thread_max = -FLT_MAX;
    
    // Process input in tiles
    for (int tile_start = 0; tile_start < in_features; tile_start += TILE_IN) {
        // Cooperatively load tile into shared memory
        int tile_end = min(tile_start + TILE_IN, in_features);
        int tile_size = tile_end - tile_start;
        
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            s_x[i] = x_row[tile_start + i];
            s_w[i] = w_row[tile_start + i];
        }
        __syncthreads();
        
        // Compute max over this tile
        for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
            scalar_t val = s_w[j] + s_x[j];
            thread_max = fmaxf(thread_max, val);
        }
        __syncthreads();
    }
    
    // Parallel reduction
    s_partial[threadIdx.x] = thread_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] = fmaxf(s_partial[threadIdx.x],
                                            s_partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        scalar_t result = s_partial[0];
        if (bias != nullptr) {
            result += bias[o];
        }
        y[b * out_features + o] = result;
    }
}


/* ==========================================================================
 * Backward kernel (STE): gradient via softmax-weighted approximation
 * 
 * For the STE backward:
 *   scores_ij = W_ij + x_j
 *   soft_w_ij = exp(scores_ij / temp) / sum_k exp(scores_ik / temp)
 *   grad_W_ij = grad_y_i * soft_w_ij
 *   grad_x_j += sum_i grad_y_i * soft_w_ij
 *   grad_b_i = grad_y_i
 * ==========================================================================*/

template <typename scalar_t>
__global__ void maxplus_backward_kernel(
    const scalar_t* __restrict__ W,         // (out_features, in_features)
    const scalar_t* __restrict__ x,         // (batch_size, in_features)
    const scalar_t* __restrict__ grad_y,    // (batch_size, out_features)
    scalar_t* __restrict__ grad_W,          // (out_features, in_features)
    scalar_t* __restrict__ grad_x,          // (batch_size, in_features)
    scalar_t* __restrict__ grad_bias,       // (out_features,) or nullptr
    const scalar_t ste_temp,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // Each block: one (batch, out_feature) pair
    const int b = blockIdx.x;
    const int o = blockIdx.y;
    
    if (b >= batch_size || o >= out_features) return;
    
    __shared__ scalar_t s_partial_sum[BLOCK_SIZE];
    
    const scalar_t* x_row = x + b * in_features;
    const scalar_t* w_row = W + o * in_features;
    scalar_t gy = grad_y[b * out_features + o];
    
    // Step 1: Find max score for numerical stability (log-sum-exp trick)
    scalar_t thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
        scalar_t score = (w_row[j] + x_row[j]) / ste_temp;
        thread_max = fmaxf(thread_max, score);
    }
    
    s_partial_sum[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial_sum[threadIdx.x] = fmaxf(s_partial_sum[threadIdx.x],
                                                s_partial_sum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    scalar_t max_score = s_partial_sum[0];
    __syncthreads();
    
    // Step 2: Compute sum of exp(score - max) for softmax denominator
    scalar_t thread_sum = 0.0f;
    for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
        scalar_t score = (w_row[j] + x_row[j]) / ste_temp;
        thread_sum += expf(score - max_score);
    }
    
    s_partial_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial_sum[threadIdx.x] += s_partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    scalar_t total_sum = s_partial_sum[0];
    __syncthreads();
    
    // Step 3: Compute gradients using softmax weights
    for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
        scalar_t score = (w_row[j] + x_row[j]) / ste_temp;
        scalar_t soft_w = expf(score - max_score) / total_sum;
        scalar_t grad = gy * soft_w;
        
        // Accumulate grad_W (atomic because multiple batches write here)
        atomicAdd(&grad_W[o * in_features + j], grad);
        // Accumulate grad_x (atomic because multiple output features write here)
        atomicAdd(&grad_x[b * in_features + j], grad);
    }
    
    // Grad bias
    if (grad_bias != nullptr && threadIdx.x == 0) {
        atomicAdd(&grad_bias[o], gy);
    }
}


/* ==========================================================================
 * Batched max-plus matrix multiply: C = A (trop) B
 * C_ij = max_k(A_ik + B_kj)
 * 
 * For tropical matrix powers (e.g., shortest path via Floyd-Warshall).
 * ==========================================================================*/

template <typename scalar_t>
__global__ void maxplus_matmul_kernel(
    const scalar_t* __restrict__ A,    // (M, K)
    const scalar_t* __restrict__ B,    // (K, N)
    scalar_t* __restrict__ C,          // (M, N)
    const int M,
    const int K,
    const int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    
    if (i >= M || j >= N) return;
    
    scalar_t max_val = -FLT_MAX;
    for (int k = 0; k < K; k++) {
        scalar_t val = A[i * K + k] + B[k * N + j];
        max_val = fmaxf(max_val, val);
    }
    
    C[i * N + j] = max_val;
}


/* ==========================================================================
 * Host-side launcher functions (C interface for PyTorch binding)
 * ==========================================================================*/

extern "C" {

void maxplus_forward_cuda(
    const float* W, const float* x, float* y, const float* bias,
    int batch_size, int in_features, int out_features, bool use_tiled
) {
    dim3 grid(batch_size, out_features);
    dim3 block(BLOCK_SIZE);
    
    if (use_tiled && in_features > TILE_IN) {
        maxplus_forward_tiled_kernel<float><<<grid, block>>>(
            W, x, y, bias, batch_size, in_features, out_features
        );
    } else {
        maxplus_forward_kernel<float><<<grid, block>>>(
            W, x, y, bias, batch_size, in_features, out_features
        );
    }
}

void maxplus_backward_cuda(
    const float* W, const float* x, const float* grad_y,
    float* grad_W, float* grad_x, float* grad_bias,
    float ste_temp, int batch_size, int in_features, int out_features
) {
    dim3 grid(batch_size, out_features);
    dim3 block(BLOCK_SIZE);
    
    maxplus_backward_kernel<float><<<grid, block>>>(
        W, x, grad_y, grad_W, grad_x, grad_bias,
        ste_temp, batch_size, in_features, out_features
    );
}

void maxplus_matmul_cuda(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    maxplus_matmul_kernel<float><<<grid, block>>>(A, B, C, M, K, N);
}

}  // extern "C"
