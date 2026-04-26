/*
 * Prefix-Max Scan CUDA Kernel (Hillis-Steele Parallel Scan)
 * ==========================================================
 *
 * Computes cumulative maximum along the sequence dimension:
 *   out[b, t, d] = max(x[b, 0, d], ..., x[b, t, d])
 *
 * This replaces torch.cummax (which falls back to CPU on DirectML)
 * and the pure-PyTorch Hillis-Steele scan (which uses log2(T)
 * iterations of memory-heavy cat+maximum).
 *
 * Strategy:
 *   - Each block processes one (batch, feature) pair along full time dim.
 *   - Warp-level shuffle scan for T <= 1024.
 *   - Multi-stage block scan for longer sequences (shared memory buffer).
 *
 * Supports: float32, float16 (via template).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

/* --------------------------------------------------------------------------
 * Warp-level prefix-max using __shfl_up_sync (Kogge-Stone tree)
 * -------------------------------------------------------------------------- */

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_prefix_max(scalar_t val, int lane_id) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        scalar_t other = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) {
            val = fmaxf(val, other);
        }
    }
    return val;
}

/* --------------------------------------------------------------------------
 * Block-level prefix-max scan using shared memory.
 * Each thread handles one time step.  Supports T up to MAX_THREADS_PER_BLOCK.
 * -------------------------------------------------------------------------- */

template <typename scalar_t>
__global__ void prefix_max_scan_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int B,
    const int T,
    const int D
) {
    extern __shared__ char smem_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem_raw);

    const int b = blockIdx.x;   // batch index
    const int d = blockIdx.y;   // feature index
    const int tid = threadIdx.x;

    if (b >= B || d >= D) return;

    // Base pointer for this (b, d) slice
    const scalar_t* x = input + b * T * D + d;
    scalar_t* y = output + b * T * D + d;

    // Load from global (strided by D)
    scalar_t val = (tid < T) ? x[tid * D] : scalar_t(-FLT_MAX);
    sdata[tid] = val;
    __syncthreads();

    // Intra-warp scan
    int lane = tid % WARP_SIZE;
    scalar_t warp_sum = warp_prefix_max(val, lane);
    __syncthreads();

    // Write warp aggregates to shared memory
    if (lane == WARP_SIZE - 1) {
        sdata[tid / WARP_SIZE] = warp_sum;
    }
    __syncthreads();

    // Inter-warp scan (one warp handles the warp aggregates)
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        scalar_t v = sdata[tid];
        int lid = tid % WARP_SIZE;
        scalar_t block_sum = warp_prefix_max(v, lid);
        sdata[tid] = block_sum;
    }
    __syncthreads();

    // Add preceding warp aggregate to each lane
    int warp_id = tid / WARP_SIZE;
    if (warp_id > 0) {
        scalar_t preceding = sdata[warp_id - 1];
        warp_sum = fmaxf(warp_sum, preceding);
    }

    // Store result
    if (tid < T) {
        y[tid * D] = warp_sum;
    }
}

/* --------------------------------------------------------------------------
 * Large-sequence prefix-max (T > 1024).
 * Each block handles a tile; block aggregates are written to aux buffer,
 * then a second pass adds the prefix of previous blocks.
 * -------------------------------------------------------------------------- */

template <typename scalar_t>
__global__ void prefix_max_tile_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ block_max,  // (B, D, n_tiles)
    const int B,
    const int T,
    const int D,
    const int tile_size
) {
    const int b = blockIdx.x;
    const int d = blockIdx.y;
    const int tile_id = blockIdx.z;
    const int tid = threadIdx.x;

    if (b >= B || d >= D) return;

    const int t0 = tile_id * tile_size;
    const int t1 = min(t0 + tile_size, T);
    const int local_T = t1 - t0;

    const scalar_t* x = input + b * T * D + d;
    scalar_t* y = output + b * T * D + d;

    extern __shared__ char smem_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem_raw);

    scalar_t val = (tid < local_T) ? x[(t0 + tid) * D] : scalar_t(-FLT_MAX);
    sdata[tid] = val;
    __syncthreads();

    int lane = tid % WARP_SIZE;
    scalar_t warp_agg = warp_prefix_max(val, lane);
    __syncthreads();

    if (lane == WARP_SIZE - 1) {
        sdata[tid / WARP_SIZE] = warp_agg;
    }
    __syncthreads();

    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        scalar_t v = sdata[tid];
        int lid = tid % WARP_SIZE;
        sdata[tid] = warp_prefix_max(v, lid);
    }
    __syncthreads();

    int warp_id = tid / WARP_SIZE;
    if (warp_id > 0) {
        scalar_t preceding = sdata[warp_id - 1];
        warp_agg = fmaxf(warp_agg, preceding);
    }

    if (tid < local_T) {
        y[(t0 + tid) * D] = warp_agg;
    }

    // Write tile aggregate
    if (tid == local_T - 1 || (tid == blockDim.x - 1 && local_T > 0)) {
        block_max[(b * D + d) * gridDim.z + tile_id] = warp_agg;
    }
}

/* --------------------------------------------------------------------------
 * Second pass: add previous-tile prefix to each element.
 * -------------------------------------------------------------------------- */

template <typename scalar_t>
__global__ void prefix_max_add_carry_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ tile_prefix,  // after exclusive scan on block_max
    const int B,
    const int T,
    const int D,
    const int tile_size
) {
    const int b = blockIdx.x;
    const int d = blockIdx.y;
    const int tile_id = blockIdx.z;
    const int tid = threadIdx.x;

    if (b >= B || d >= D || tile_id == 0) return;

    const int t0 = tile_id * tile_size;
    const int t1 = min(t0 + tile_size, T);
    const int local_T = t1 - t0;

    if (tid >= local_T) return;

    scalar_t* y = output + b * T * D + d;
    scalar_t carry = tile_prefix[(b * D + d) * gridDim.z + (tile_id - 1)];
    y[(t0 + tid) * D] = fmaxf(y[(t0 + tid) * D], carry);
}

/* ==========================================================================
 * Host launchers
 * ========================================================================== */

extern "C" {

void prefix_max_scan_cuda(
    const float* input,
    float* output,
    int B,
    int T,
    int D
) {
    dim3 grid(B, D);
    int threads = min(T, MAX_THREADS_PER_BLOCK);
    int smem = threads * sizeof(float);

    if (T <= MAX_THREADS_PER_BLOCK) {
        prefix_max_scan_kernel<float><<<grid, threads, smem>>>(
            input, output, B, T, D
        );
    } else {
        int tile_size = MAX_THREADS_PER_BLOCK;
        int n_tiles = (T + tile_size - 1) / tile_size;
        dim3 grid2(B, D, n_tiles);
        int smem2 = tile_size * sizeof(float);

        // Allocate aux buffer on device (caller must free)
        float* d_block_max = nullptr;
        cudaMalloc(&d_block_max, B * D * n_tiles * sizeof(float));

        prefix_max_tile_kernel<float><<<grid2, tile_size, smem2>>>(
            input, output, d_block_max, B, T, D, tile_size
        );

        // Scan tile aggregates (in-place on d_block_max)
        prefix_max_scan_kernel<float><<<grid, threads, smem>>>(
            d_block_max, d_block_max, B, n_tiles, D
        );

        // Add carry
        prefix_max_add_carry_kernel<float><<<grid2, tile_size>>>(
            output, d_block_max, B, T, D, tile_size
        );

        cudaFree(d_block_max);
    }
}

} // extern "C"
