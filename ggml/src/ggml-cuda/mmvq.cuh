#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Whether the multi-column split-load MMVQ kernel handles this case; it also supports gate/GLU fusion for ncols_dst > 1.
bool ggml_cuda_mmvq_s8_usable(ggml_type type, int cc, int64_t ncols_dst, bool has_ids);

// Whether the few-column kernel can fuse a gate/up pair whose weights use different quant types.
bool ggml_cuda_mmvq_s8_can_fuse_mixed(const ggml_tensor * up_w, const ggml_tensor * gate_w, int64_t ncols_dst);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// src1_q8_1: optional, src1 already quantized by the caller (e.g. a per-graph q8_1 activation cache), else src1 is
// quantized here. It must hold exactly what this function would produce: quantize_row_q8_1_cuda() of src1 with
// ne0 = GGML_PAD(ne10, MATRIX_ROW_PADDING), i.e. ne11*ne12*ne13 packed rows of ne0/QK8_1 block_q8_1 each.
// Works with or without fusion (also the few-column gate/up(+GLU) and bias fusions for ncols_dst 2..8).
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr,
    const void * src1_q8_1 = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
