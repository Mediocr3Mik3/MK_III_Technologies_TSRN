/*
 * Fused Tropical SSM State Recurrence CUDA Kernel
 * ================================================
 *
 * Computes the max-plus linear recurrence used in TropicalSSM:
 *   G_t = max( G_{t-1},  u_t - t * A_c )
 *   h_t = t * A_c + G_t
 * where A_c <= 0 (clamped decay), and max is element-wise.
 *
 * This is a selective prefix-max scan over (u_t - t * A_c).
 * Fusing the t-scale, init, and scan into one kernel avoids
 * materialising the (B, T+1, ds) intermediate G tensor.
 *
 * Strategy:
 *   - Each thread block processes one (batch, state_dim) pair.
 *   - Warp-shuffle prefix max within block.
 *   - Multi-block pass for long sequences (T > 1024).
 *
 * Inputs:
 *   u:   (B, T, ds)  selective input (gate_B * B_proj(x))
 *   A_c: (ds,)       non-positive decay vector
 * Outputs:
 *   h:   (B, T, ds)  state
 */

#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define MAX_THREADS 1024

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_max(scalar_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        scalar_t other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_prefix_max(scalar_t val, int lane) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        scalar_t other = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane >= offset) val = fmaxf(val, other);
    }
    return val;
}

/* --------------------------------------------------------------------------
 * Single-pass fused SSM recurrence for T <= 1024.
 * Each block = one (b, d).  Threads iterate over t in steps of blockDim.x.
 * -------------------------------------------------------------------------- */
template <typename scalar_t>
__global__ void tropical_ssm_fused_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ A_c,
    scalar_t* __restrict__ h,
    int B, int T, int ds
) {
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    int b = blockIdx.x;
    int d = blockIdx.y;
    int tid = threadIdx.x;

    if (b >= B || d >= ds) return;

    scalar_t a = A_c[d];  // A_c <= 0

    // We process the sequence in chunks that fit in one block.
    // For T <= MAX_THREADS, one chunk covers everything.
    scalar_t running_max = a;  // G_init = A_c (the t=0 init value)

    for (int t_base = 0; t_base < T; t_base += blockDim.x) {
        int t = t_base + tid;
        scalar_t val = (t < T) ? u[b * T * ds + t * ds + d] - t * a : scalar_t(-FLT_MAX);
        sdata[tid] = val;
        __syncthreads();

        // Prefix max within this chunk
        int lane = tid % WARP_SIZE;
        scalar_t warp_agg = warp_prefix_max(val, lane);
        __syncthreads();

        if (lane == WARP_SIZE - 1) sdata[tid / WARP_SIZE] = warp_agg;
        __syncthreads();

        if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
            scalar_t v = sdata[tid];
            sdata[tid] = warp_prefix_max(v, tid % WARP_SIZE);
        }
        __syncthreads();

        int warp_id = tid / WARP_SIZE;
        if (warp_id > 0) {
            scalar_t preceding = sdata[warp_id - 1];
            warp_agg = fmaxf(warp_agg, preceding);
        }
        __syncthreads();

        // Combine with running_max from previous chunks
        scalar_t g = fmaxf(running_max, warp_agg);
        if (t < T) {
            h[b * T * ds + t * ds + d] = t * a + g;
        }

        // Update running_max for next chunk
        if (tid == blockDim.x - 1 || (t == T - 1)) {
            running_max = g;
        }
        __syncthreads();
    }
}

/* --------------------------------------------------------------------------
 * Multi-block variant for T > 1024.
 *
 * The recurrence factors as:
 *      h_t = t*a + max_{s<=t} (u_s - s*a)
 * Define  v_t := u_t - t*a.  Then  h_t = t*a + prefix_max(v)_t.
 *
 * Tile pass:   each tile computes prefix_max(v) within its own range and
 *              writes the FULL tile aggregate (i.e. prefix_max(v)_{t1-1})
 *              to ``tile_aggs[..., tile_id]``.
 *
 * Tile scan:   scan tile_aggs (inclusive) with prefix_max_scan_kernel from
 *              prefix_max_scan.cu, producing a per-tile carry value.
 *
 * Carry pass:  for tile_id >= 1, take the carry = scan(tile_aggs)[tile_id-1],
 *              and update h_t = t*a + max(prefix_max_v_local_t, carry).
 *              Equivalently: h_t = max(h_t, t*a + carry).
 * -------------------------------------------------------------------------- */
template <typename scalar_t>
__global__ void tropical_ssm_fused_tiles_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ A_c,
    scalar_t* __restrict__ h,
    scalar_t* __restrict__ tile_aggs,  // (B, ds, n_tiles) — stored as scalar_t
    int B, int T, int ds, int tile_size
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int tile_id = blockIdx.z;
    int tid = threadIdx.x;

    if (b >= B || d >= ds) return;

    int t0 = tile_id * tile_size;
    int t1 = min(t0 + tile_size, T);
    int local_T = t1 - t0;
    if (local_T <= 0) return;

    scalar_t a = A_c[d];
    int t = t0 + tid;
    scalar_t val = (tid < local_T) ? (u[b * T * ds + t * ds + d] - t * a)
                                    : scalar_t(-FLT_MAX);

    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    sdata[tid] = val;
    __syncthreads();

    int lane = tid % WARP_SIZE;
    scalar_t warp_agg = warp_prefix_max(val, lane);
    __syncthreads();

    if (lane == WARP_SIZE - 1) sdata[tid / WARP_SIZE] = warp_agg;
    __syncthreads();

    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        scalar_t v = sdata[tid];
        sdata[tid] = warp_prefix_max(v, tid % WARP_SIZE);
    }
    __syncthreads();

    int warp_id = tid / WARP_SIZE;
    if (warp_id > 0) {
        warp_agg = fmaxf(warp_agg, sdata[warp_id - 1]);
    }

    if (tid < local_T) {
        // store h_t = t*a + prefix_max_v_local_t  (carry to be folded in later)
        h[b * T * ds + t * ds + d] = t * a + warp_agg;
    }

    // Last valid thread in tile writes the FULL tile aggregate
    if (tid == local_T - 1) {
        tile_aggs[(b * ds + d) * gridDim.z + tile_id] = warp_agg;
    }
}

template <typename scalar_t>
__global__ void tropical_ssm_add_carry_kernel(
    scalar_t* __restrict__ h,
    const scalar_t* __restrict__ A_c,
    const scalar_t* __restrict__ tile_prefix_inclusive,
    int B, int T, int ds, int tile_size
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int tile_id = blockIdx.z;
    int tid = threadIdx.x;

    if (b >= B || d >= ds || tile_id == 0) return;

    int t0 = tile_id * tile_size;
    int t1 = min(t0 + tile_size, T);
    int local_T = t1 - t0;
    if (tid >= local_T) return;

    int t = t0 + tid;
    scalar_t a = A_c[d];

    // Carry from previous tile, in v-space (prefix_max of u_s - s*a for s < t0)
    scalar_t carry_v = tile_prefix_inclusive[(b * ds + d) * gridDim.z + (tile_id - 1)];

    int idx = b * T * ds + t * ds + d;
    // h was written as t*a + prefix_max_v_local_t.
    // True h_t = t*a + max(carry_v, prefix_max_v_local_t).
    // So: h_t = max(h_t, t*a + carry_v).
    scalar_t shifted_carry = t * a + carry_v;
    h[idx] = fmaxf(h[idx], shifted_carry);
}

/* ==========================================================================
 * Host launcher
 * ========================================================================== */
extern "C" {

void tropical_ssm_fused_cuda(
    const float* u,
    const float* A_c,
    float* h,
    int B,
    int T,
    int ds
) {
    dim3 grid(B, ds);
    int threads = min(T, MAX_THREADS);
    int smem = threads * sizeof(float);

    tropical_ssm_fused_kernel<float><<<grid, threads, smem>>>(
        u, A_c, h, B, T, ds
    );
}

} // extern "C"
