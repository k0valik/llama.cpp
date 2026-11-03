#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#define GGML_CUDA_CONV_STATE_MAX_SNAPSHOTS 8
#define GGML_CUDA_CONV_STATE_MAX_T 16

struct ggml_cuda_conv_state_snapshots {
    float * dst[GGML_CUDA_CONV_STATE_MAX_SNAPSHOTS]; // contiguous [n_state, nr] destination (one sequence)
    int     off[GGML_CUDA_CONV_STATE_MAX_SNAPSHOTS]; // window start of the snapshot
    int     n;
};

struct ggml_cuda_conv_state_fused_args {
    const char    * st;        // conv state: element (j, c) at st + row*st_row_nb + j*st_nb0 + c*st_nb1
    const int32_t * st_row;    // optional device row index (elided get_rows), nullptr -> row 0
    int64_t st_row_nb, st_nb0, st_nb1;
    const char    * x;         // new inputs: element (t, c) at x + t*x_nb0 + c*x_nb1
    int64_t x_nb0, x_nb1;
    const float   * w;         // conv weight [d_conv, nr], row stride w_s1 (floats)
    int64_t w_s1;
    float         * y;         // silu(conv) output [nr, n_t], column stride y_s1 (floats)
    int64_t y_s1;
    ggml_cuda_conv_state_snapshots snaps;
    int nr, n_t, d_conv;
};

// returns false when the shape is not supported (caller falls back to the unfused ops)
bool ggml_cuda_op_ssm_conv_state_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state_fused_args & args);
