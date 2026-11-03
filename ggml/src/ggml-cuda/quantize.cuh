#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// silu(a)*b quantized to the MMQ q8_1 layout (a, b: ne1 rows of ne00 floats with row strides s_a, s_b; ne0: padded row length)
void quantize_mmq_q8_1_silu_mul_cuda(
        const float * a, const float * b, void * vy, ggml_type type_src0,
        int64_t ne00, int64_t s_a, int64_t s_b, int64_t ne0, int64_t ne1, cudaStream_t stream);

// silu(z) * (rms_norm(o)*w) quantized to the MMQ q8_1 layout, the rms norm groups are the 128 value blocks of the rows
// (o, z: ne1 contiguous rows of ne00 floats, w: 128 floats). false if unsupported (nothing launched).
bool quantize_mmq_q8_1_rms_silu_mul_cuda(
        const float * o, const float * w, const float * z, void * vy, ggml_type type_src0, float eps,
        int64_t ne00, int64_t ne0, int64_t ne1, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);
