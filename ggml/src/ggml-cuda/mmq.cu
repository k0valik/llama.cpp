#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_q_case<GGML_TYPE_IQ1_M>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_type type, const ggml_prec prec_requested, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = prec_requested;
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

// Tile width in ne11 (dense) / ne12 (MoE) direction with the fewest tiles that fits into shared memory, 0 if there is none.
static int ggml_cuda_mmq_get_J_best(
        const ggml_type type, const int64_t ncols, const bool fallback, const int cc, const size_t smpbo, const ggml_prec prec_src1,
        int * nthreads_best = nullptr) {
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (ncols + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            if (nthreads_best) {
                *nthreads_best = config.nthreads;
            }
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    // Fused gate/up/GLU: src0 is the up weight, dst the GLU output.
    const ggml_tensor * gate = fusion ? fusion->gate : nullptr;
    ggml_cuda_mm_fusion_args_device fusion_device = {};
    if (fusion) {
        GGML_ASSERT(gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale == !fusion->gate_scale);
        GGML_ASSERT(gate->type == src0->type && ggml_are_same_shape(gate, src0) && ggml_are_same_stride(gate, src0));
        // NVFP4 weight scales (only applied for NVFP4): one value per tensor, or one per expert with ids.
        for (const ggml_tensor * scale : { fusion->x_scale, fusion->gate_scale }) {
            GGML_ASSERT(!scale || (src0->type == GGML_TYPE_NVFP4 && scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) &&
                ggml_nelements(scale) == (ids ? src0->ne[2] : 1)));
        }
        fusion_device.gate       = gate->data;
        fusion_device.x_scale    = fusion->x_scale    ? fusion->x_scale->data    : nullptr;
        fusion_device.gate_scale = fusion->gate_scale ? fusion->gate_scale->data : nullptr;
        fusion_device.glu_op     = fusion->glu_op;
        fusion_device.glu_limit  = fusion->glu_limit;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 (or the fused gate weight) is a temporary compute buffer, clear any potential padding.
    for (const ggml_tensor * t : { src0, gate }) {
        if (t && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_t size_data  = ggml_nbytes(t);
            const size_t size_alloc = ggml_backend_buffer_get_alloc_size(t->buffer, t);
            if (size_alloc > size_data) {
                GGML_ASSERT(ggml_is_contiguously_allocated(t));
                GGML_ASSERT(!t->view_src);
                CUDA_CHECK(cudaMemsetAsync((char *) t->data + size_data, 0, size_alloc - size_data, stream));
            }
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);

    // With fusion dst is the GLU output, not a MUL_MAT; the src1 precision requested by both fused MUL_MATs is in the fusion args.
    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0->type, gate ? fusion->prec_src1 : ggml_prec_get_src(dst, 1), cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    int J_best        = 0;
    int nthreads_best = 0;
    {
        int64_t ncols_opt = ne11;
        if (ids) {
            const int64_t n_expert_used = ids->ne[0];
            ncols_opt = ne12;

            // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
            ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
        }

        J_best = ggml_cuda_mmq_get_J_best(src0->type, ncols_opt, fallback, cc, smpbo, prec_src1, &nthreads_best);
    }
    GGML_ASSERT(J_best > 0);

    // A tile of size J can read in at most J - 1 extra columns.
    // For simplicity, round up the padding of a full tile to a multiple of the number of bytes that nthreads can load in parallel.
    const size_t src1_load_chunk_size = nthreads_best * sizeof(int);
    const size_t src1_q8_1_padding = ((J_best * sizeof(block_q8_1_mmq) + src1_load_chunk_size - 1) / src1_load_chunk_size)
        * src1_load_chunk_size;

    if (!ids) {
        const int ntj = (ne1 + J_best - 1) / J_best;

        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ntj, J_best, fusion_device};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + J_best-1); // Needs to be padded for unconditional memory access.

    const int ntj = (ne_get_rows + J_best - 1) / J_best + ne02;
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), ntj);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), tile_spec.get(),
            ne02, ne12, n_expert_used, ne11, J_best, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), tile_spec.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02,    1, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ntj, J_best, fusion_device};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
}

// Match the normal MMQ launch selection and its padded, unconditional src1 loads.
static size_t ggml_cuda_mmq_q8_padding(const ggml_tensor * src0, const int64_t ncols, const int cc, int & J) {
    int nthreads = 0;
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    J = ggml_cuda_mmq_get_J_best(src0->type, ncols, ggml_cuda_mmq_needs_fallback(src0->ne[1]),
            cc, smpbo, GGML_PREC_Q8, &nthreads);
    if (J == 0) {
        return 0;
    }
    const size_t chunk = nthreads * sizeof(int);
    return ((J * sizeof(block_q8_1_mmq) + chunk - 1) / chunk) * chunk;
}

// dst = src0 * (silu(a)*b): the gated activation is computed while quantizing src1 for MMQ (prefill SwiGLU etc.).
// a, b: contiguous F32 with ne00*ne11 elements each, dst: contiguous F32 [ne01, ne11], src0 2D. Q8_1 src1 only.
bool ggml_cuda_mul_mat_q_silu_mul(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * a, const ggml_tensor * b, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = dst->ne[1];
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || dst->ne[0] != ne01 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ne00 % 4 != 0 ||
            src0->nb[0] != ggml_type_size(src0->type) || src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4 ||
            a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
            ggml_nelements(a) != ne00*ne11 || ggml_nelements(b) != ne00*ne11 ||
            ggml_cuda_mmq_get_prec_src1(src0->type, ggml_prec_get_src(dst, 1), cc) != GGML_PREC_Q8 ||
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    int J;
    const size_t padding = ggml_cuda_mmq_q8_padding(src0, ne11, cc, J);
    if (J == 0) {
        return false;
    }

    const size_t nbytes_src1_q8_1 = ne11*ne10_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ + padding;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    quantize_mmq_q8_1_silu_mul_cuda((const float *) a->data, (const float *) b->data, src1_q8_1.get(), src0->type,
        ne00, /*s_a =*/ ne00, /*s_b =*/ ne00, ne10_padded, ne11, stream);
    CUDA_CHECK(cudaGetLastError());

    const int64_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t s1  = dst->nb[1] / sizeof(float);
    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));

    const mmq_args args = {
        (const char *) src0->data, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, (float *) dst->data,
        nullptr,
        ne00, ne01, ne11, s01, ne11, s1,
        1, 1, 0, s12, 0,
        1, 1, 0, s12, 0,
        ne11, J};
    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, GGML_PREC_Q8);
    return true;
}

// Two matmuls with the same src1 (e.g. the FFN gate and up projections in prefill): src1 is quantized only once.
bool ggml_cuda_mul_mat_q_shared_src1(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_tensor * dst_a, ggml_tensor * dst_b) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const ggml_tensor * dsts[2] = { dst_a, dst_b };
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) || src1->ne[2] != 1 || src1->ne[3] != 1 || ne10 % 4 != 0) {
        return false;
    }
    for (const ggml_tensor * dst : dsts) {
        const ggml_tensor * src0 = dst->src[0];
        if (dst->src[1] != src1 || src0->ne[0] != ne10 || src0->ne[2] != 1 || src0->ne[3] != 1 || dst->ne[1] != ne11 ||
                dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || src0->nb[0] != ggml_type_size(src0->type) ||
                src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4 ||
                ggml_cuda_mmq_get_prec_src1(src0->type, ggml_prec_get_src(dst, 1), cc) != GGML_PREC_Q8 ||
                ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
                mmq_get_q8_1_ds_layout(src0->type) != mmq_get_q8_1_ds_layout(dst_a->src[0]->type)) {
            return false;
        }
    }

    cudaStream_t stream = ctx.stream();

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    int J[2];
    size_t padding = 0;
    for (int i = 0; i < 2; ++i) {
        padding = std::max(padding, ggml_cuda_mmq_q8_padding(dsts[i]->src[0], ne11, cc, J[i]));
        if (J[i] == 0) {
            return false;
        }
    }
    const size_t nbytes_src1_q8_1 = ne11*ne10_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ + padding;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), dst_a->src[0]->type, ne10,
        src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float), src1->nb[3]/sizeof(float), ne10_padded, ne11, 1, 1, stream);
    CUDA_CHECK(cudaGetLastError());

    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    for (int i = 0; i < 2; ++i) {
        const ggml_tensor * dst = dsts[i];
        const ggml_tensor * src0 = dst->src[0];
        const mmq_args args = {
            (const char *) src0->data, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, (float *) dst->data,
            nullptr,
            ne10, src0->ne[1], ne11, (int64_t) (src0->nb[1] / ggml_type_size(src0->type)), ne11, (int64_t) (dst->nb[1] / sizeof(float)),
            1, 1, 0, s12, 0,
            1, 1, 0, s12, 0,
            ne11, J[i]};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, GGML_PREC_Q8);
    }
    return true;
}

// dst = src0 * (silu(z) * rms_norm(o)*w) with the gated rms norm (128 value groups) fused into the src1 quantization.
// o, z: contiguous F32 with ne00*ne11 elements each, w: 128 F32, dst: contiguous F32 [ne01, ne11], src0 2D.
bool ggml_cuda_mul_mat_q_rms_silu_mul(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * o, const ggml_tensor * w,
        const float eps, const ggml_tensor * z, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = dst->ne[1];
    const ggml_type t  = src0->type;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || dst->ne[0] != ne01 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ne00 % QK8_1_MMQ != 0 ||
            src0->nb[0] != ggml_type_size(t) || t == GGML_TYPE_MXFP4 || t == GGML_TYPE_NVFP4 ||
            mmq_get_q8_1_ds_layout(t) == MMQ_Q8_1_DS_LAYOUT_D2S6 ||
            o->type != GGML_TYPE_F32 || z->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(o) || !ggml_is_contiguous(z) || !ggml_is_contiguous(w) ||
            ggml_nelements(o) != ne00*ne11 || ggml_nelements(z) != ne00*ne11 || ggml_nelements(w) != QK8_1_MMQ ||
            ggml_cuda_mmq_get_prec_src1(src0->type, ggml_prec_get_src(dst, 1), cc) != GGML_PREC_Q8 ||
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    int J;
    const size_t padding = ggml_cuda_mmq_q8_padding(src0, ne11, cc, J);
    if (J == 0) {
        return false;
    }

    const size_t nbytes_src1_q8_1 = ne11*ne10_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ + padding;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    if (!quantize_mmq_q8_1_rms_silu_mul_cuda((const float *) o->data, (const float *) w->data, (const float *) z->data,
            src1_q8_1.get(), t, eps, ne00, ne10_padded, ne11, stream)) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());

    const int64_t s01 = src0->nb[1] / ggml_type_size(t);
    const int64_t s1  = dst->nb[1] / sizeof(float);
    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));

    const mmq_args args = {
        (const char *) src0->data, t, (const int *) src1_q8_1.ptr, nullptr, nullptr, (float *) dst->data,
        nullptr,
        ne00, ne01, ne11, s01, ne11, s1,
        1, 1, 0, s12, 0,
        1, 1, 0, s12, 0,
        ne11, J};
    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, GGML_PREC_Q8);
    return true;
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        case GGML_TYPE_IQ1_M:
            // only tested with the MMA code path on Blackwell
            mmq_supported = GGML_CUDA_CC_IS_NVIDIA(cc) && blackwell_mma_available(cc);
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

bool ggml_cuda_mmq_fusion_is_efficient(
        const ggml_type type, const ggml_prec prec_requested, const int64_t nrows_x, const int64_t ncols, const int64_t nchannels, const int device) {
    const int    cc    = ggml_cuda_info().devices[device].cc;
    const int    nsm   = ggml_cuda_info().devices[device].nsm;
    const size_t smpbo = ggml_cuda_info().devices[device].smpbo;

    // Same tile width as ggml_cuda_mul_mat_q; the fused kernel has no fallback and writes I/2 output rows per tile.
    const bool      fallback  = false;
    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(type, prec_requested, cc);
    const int       J_best    = ggml_cuda_mmq_get_J_best(type, ncols, fallback, cc, smpbo, prec_src1);
    if (J_best == 0) {
        return false;
    }
    const int64_t nrows_tile = ggml_cuda_mmq_get_config(type, J_best, fallback, cc, prec_src1).I / 2;
    const int64_t ntiles     = (ncols + J_best - 1)/J_best * ((nrows_x + nrows_tile - 1)/nrows_tile) * nchannels;
    // Same condition as launch_mul_mat_q uses to run the stream-k kernel with one block per tile.
    const int64_t tiles_nwaves             = (ntiles + nsm - 1) / nsm;
    const int64_t tiles_efficiency_percent = 100 * ntiles / (nsm*tiles_nwaves);
    return tiles_efficiency_percent >= 90;
}
