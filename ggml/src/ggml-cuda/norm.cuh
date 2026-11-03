#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// y = rms_norm(a [+ b]) * w (rows of ncols), optionally writing a + b to sum_out and y as q8_1 (row stride q8_row_blocks)
// b / sum_out / q8 may be nullptr; returns false for unsupported shapes
bool ggml_cuda_add_rms_norm_mul(ggml_backend_cuda_context & ctx, const float * a, const float * b, float * sum_out,
                                const float * w, float * y, void * q8, int ncols, int nrows,
                                int64_t s_a, int64_t s_b, int64_t s_sum, int64_t s_y, int64_t q8_row_blocks, float eps,
                                bool no_alias); // no_alias: sum_out/y overlap neither a nor b (allows the segmented kernel)
