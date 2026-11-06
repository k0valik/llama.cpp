#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = col < d_conv - 1 + local_n_t ? x_block[row * stride_x + col] : 0.0f;
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

template <int n_state>
static __device__ __forceinline__ float * conv_state_snapshot_dst(const ggml_cuda_conv_state_snapshots & snaps, const int s, const int c) {
    int64_t row_offset = 0;
    if (snaps.rows) {
        const int32_t row = snaps.rows[s];
        if (row < 0 || row >= snaps.n_rows) {
            return nullptr;
        }
        row_offset = (int64_t) row * snaps.row_stride;
    }
    return snaps.dst[s] + row_offset + (int64_t) c * n_state;
}

// Fused short-sequence conv-state path (see ggml_cuda_conv_state_fused in ggml-cuda.cu):
//   [get_rows(conv states)] -> concat(state, x^T) -> K x cpy(snapshot window -> cache) -> ssm_conv -> silu
// One thread per channel builds the (d_conv - 1 + n_t) window in shared memory, writes the snapshots and the
// silu'd conv outputs. Every thread only touches its own channel in every buffer, so reading the state row
// and writing snapshot rows of the same cache in one kernel is race-free (single sequence only).
template <int d_conv, int max_t>
static __global__ void ssm_conv_state_fused_f32(
        const char * st, const int32_t * st_row, const int64_t st_row_nb, const int64_t st_nb0, const int64_t st_nb1,
        const char * x, const int64_t x_nb0, const int64_t x_nb1,
        const float * w, const int64_t w_s1,
        float * y, const int64_t y_s1,
        const ggml_cuda_conv_state_snapshots snaps, const int nr, const int n_t) {
    constexpr int n_state = d_conv - 1;
    constexpr int n_win   = n_state + max_t;

    extern __shared__ float win_smem[];
    const int tid = threadIdx.x;
    const int c   = blockIdx.x * blockDim.x + tid;

    float wr[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wr[j] = c < nr ? w[c * w_s1 + j] : 0.0f;
    }

    ggml_cuda_pdl_lc();

    if (c >= nr) {
        ggml_cuda_pdl_sync();
        return;
    }

    // the state row must not be read before the wait: an earlier node of the same graph may still be writing it
    // (e.g. the scale_inplace that zeroes the row of a new sequence)
    ggml_cuda_pdl_sync();

    float win[n_win];
    {
        const char * st_c = st + (st_row ? (int64_t) st_row[0] * st_row_nb : 0) + c * st_nb1;
#pragma unroll
        for (int j = 0; j < n_state; ++j) {
            win[j] = *(const float *) (st_c + j * st_nb0);
        }
    }
#pragma unroll
    for (int t = 0; t < max_t; ++t) {
        win[n_state + t] = t < n_t ? *(const float *) (x + t * x_nb0 + c * x_nb1) : 0.0f;
    }

    // snapshots: window [off, off + n_state) -> dst[c * n_state + j]; off is runtime, so go through smem
#pragma unroll
    for (int j = 0; j < n_win; ++j) {
        win_smem[j * blockDim.x + tid] = win[j];
    }
    for (int s = 0; s < snaps.n; ++s) {
        const int off = snaps.off[s];
        float * dst = conv_state_snapshot_dst<n_state>(snaps, s, c);
        if (dst) {
#pragma unroll
            for (int j = 0; j < n_state; ++j) {
                dst[j] = win_smem[(off + j) * blockDim.x + tid];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < max_t; ++t) {
        if (t < n_t) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                sumf += win[t + j] * wr[j];
            }
            y[t * y_s1 + c] = ggml_cuda_op_silu_single(sumf);
        }
    }
}

// Long-sequence variant of the fused conv-state path (prefill): the grid also splits the tokens (chunks of
// tchunk per block row); each thread streams its channel over the chunk with a sliding window. The snapshot windows
// are read straight from the state / new inputs by the first token chunk.
template <int d_conv, int tchunk>
static __global__ void ssm_conv_state_fused_long_f32(
        const char * st, const int32_t * st_row, const int64_t st_row_nb, const int64_t st_nb0, const int64_t st_nb1,
        const char * x, const int64_t x_nb0, const int64_t x_nb1,
        const float * w, const int64_t w_s1,
        float * y, const int64_t y_s1,
        const ggml_cuda_conv_state_snapshots snaps, const int nr, const int n_t) {
    constexpr int n_state = d_conv - 1;
    const int c  = blockIdx.x * blockDim.x + threadIdx.x;
    const int t0 = blockIdx.y * tchunk;

    ggml_cuda_pdl_lc();

    float wr[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wr[j] = c < nr ? w[c * w_s1 + j] : 0.0f;
    }

    ggml_cuda_pdl_sync();

    if (c >= nr) {
        return;
    }

    const char * st_c = st + (st_row ? (int64_t) st_row[0] * st_row_nb : 0) + c * st_nb1;
    const char * x_c  = x + c * x_nb1;
    // window position p: p < n_state -> state, else new input p - n_state
    auto win = [&](const int p) -> float {
        return p < n_state ? *(const float *) (st_c + p * st_nb0) : *(const float *) (x_c + (int64_t) (p - n_state) * x_nb0);
    };

    // the window start (which may be the state) is read before any snapshot is written: a snapshot row may alias
    // the state row. Snapshots are written by the first token chunk, each reading its window before writing.
    float h[n_state];
#pragma unroll
    for (int j = 0; j < n_state; ++j) {
        h[j] = win(t0 + j);
    }

    if (blockIdx.y == 0) {
        float v[GGML_CUDA_CONV_STATE_MAX_SNAPSHOTS][n_state];
        for (int s = 0; s < snaps.n; ++s) {
#pragma unroll
            for (int j = 0; j < n_state; ++j) {
                v[s][j] = win(snaps.off[s] + j);
            }
        }
        for (int s = 0; s < snaps.n; ++s) {
            float * dst = conv_state_snapshot_dst<n_state>(snaps, s, c);
            if (dst) {
#pragma unroll
                for (int j = 0; j < n_state; ++j) {
                    dst[j] = v[s][j];
                }
            }
        }
    }
    const int t1 = min(n_t, t0 + tchunk);
#pragma unroll 8
    for (int t = t0; t < t1; ++t) {
        const float xt = *(const float *) (x_c + (int64_t) t * x_nb0);
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < n_state; ++j) {
            sumf += h[j] * wr[j];
        }
        sumf += xt * wr[n_state];
        y[(int64_t) t * y_s1 + c] = ggml_cuda_op_silu_single(sumf);
#pragma unroll
        for (int j = 0; j < n_state - 1; ++j) {
            h[j] = h[j + 1];
        }
        h[n_state - 1] = xt;
    }
}

bool ggml_cuda_op_ssm_conv_state_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state_fused_args & a) {
    const int threads = 128;
    const int nblocks = (a.nr + threads - 1) / threads;
    cudaStream_t stream = ctx.stream();

    auto launch = [&](auto NC, auto MT) {
        constexpr int kNC = decltype(NC)::value;
        constexpr int kMT = decltype(MT)::value;
        const size_t smem = (size_t) threads * (kNC - 1 + kMT) * sizeof(float);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(dim3(nblocks), dim3(threads), smem, stream);
        ggml_cuda_kernel_launch(ssm_conv_state_fused_f32<kNC, kMT>, launch_params,
            a.st, a.st_row, a.st_row_nb, a.st_nb0, a.st_nb1, a.x, a.x_nb0, a.x_nb1, a.w, a.w_s1, a.y, a.y_s1,
            a.snaps, a.nr, a.n_t);
    };
    if (a.n_t > GGML_CUDA_CONV_STATE_MAX_T) {
        constexpr int tchunk = 32;
        const dim3 grid(nblocks, (a.n_t + tchunk - 1) / tchunk);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, dim3(threads), 0, stream);
        auto launch_long = [&](auto NC) {
            ggml_cuda_kernel_launch(ssm_conv_state_fused_long_f32<decltype(NC)::value, tchunk>, lp,
                a.st, a.st_row, a.st_row_nb, a.st_nb0, a.st_nb1, a.x, a.x_nb0, a.x_nb1, a.w, a.w_s1, a.y, a.y_s1,
                a.snaps, a.nr, a.n_t);
        };
        switch (a.d_conv) {
            case 3: launch_long(std::integral_constant<int, 3>{}); return true;
            case 4: launch_long(std::integral_constant<int, 4>{}); return true;
            default: return false;
        }
    }
    auto launch_nc = [&](auto NC) {
        if (a.n_t <= 4) {
            launch(NC, std::integral_constant<int, 4>{});
        } else if (a.n_t <= 8) {
            launch(NC, std::integral_constant<int, 8>{});
        } else {
            launch(NC, std::integral_constant<int, 16>{});
        }
    };
    switch (a.d_conv) {
        case 3: launch_nc(std::integral_constant<int, 3>{}); return true;
        case 4: launch_nc(std::integral_constant<int, 4>{}); return true;
        default: return false;
    }
}
