#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5) || defined(RDNA4_M)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        cc = ggml_cuda_highest_compiled_arch(cc);
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        if (ggml_cuda_mmvq_s8_usable(type, cc, ne11, false)) {
            return true; // few-column split-load kernel, 2..8 columns
        }
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3) || defined(RDNA4_M)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false, bool split_k = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
                return 4;
            case 2:
            case 3:
            case 4:
                // 4 warps split K 16 blocks at a time, so K = 5120 (20 blocks) leaves 3 warps idle at the barrier.
                // Unless split_k is requested, one warp per block (4 rows, see calc_rows_per_block) keeps the K loop dense.
                return split_k ? 4 : 1;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC, false, false, /*split_k =*/ true);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
                return table_id == MMVQ_PARAMETERS_GENERIC && nwarps == 1 ? 4 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool split_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters, split_k)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters, split_k);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            uint32_t(row0 + threadIdx.x) < nrows_x) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + min(uint32_t(i), nrows_x - 1 - uint32_t(row0))*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                // Partial final blocks must still participate in every reduction and barrier.
                const uint32_t row = min(uint32_t(i), nrows_x - 1 - uint32_t(row0));
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + row*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + row*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && uint32_t(row0 + i) < nrows_x) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Few-column (ncols_dst 2..4, 5..8: mul_mat_vec_q_s8w) MMVQ kernel with split weight load/decode, NVIDIA only (mul_mat_vec_q_s8).
// Decoders: one thread handles one 32-weight slice s (0..7) of a 256-weight super block, i.e. exactly the weights that
// pair with q8_1 block 8*kbx + s. load() only issues the (few, small) global loads of the raw bits so the kernel can put
// them in flight one iteration ahead; decode() turns them into 8 ints of signed int8 weights in q8_1 order; fin() returns
// the float contribution of one slice before the activation scale d8, with the same integer rounding of the sub-block
// scales as the corresponding vec_dot_*_q8_1 function (types with a min/offset term also get the sums of the activations
// of ints 0..3 / 4..7, needs_ysum). The vec_dot functions in vecdotq.cuh (MMQ, other MMVQ paths) are unchanged.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_MMVQ_S8_AVAILABLE

// prmt.b32 with the default mode: selector nibbles with bit 3 set replicate the sign bit of the selected byte.
static __device__ __forceinline__ uint32_t mmvq_s8_prmt(const uint32_t a, const uint32_t b, const uint32_t s) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}

// Negate the bytes of g (every byte in [1, 127]) whose bit is set in the 4-bit value n:
// m1 has 0x01 in the bytes to negate, (g - m1) ^ 0xFF == -g for those bytes and never borrows because g >= 1.
static __device__ __forceinline__ int mmvq_s8_sign4(const uint32_t g, const uint32_t n) {
    const uint32_t m1 = (n * 0x00204081u) & 0x01010101u;
    return (int) ((g - m1) ^ (m1 * 0xFFu));
}

// Weight loads of the few-column kernel. Every weight byte is used exactly once, so it is loaded evict-first
// ("cache streaming", ld.global.cs; measured 0-3% faster than plain loads on the model shapes).
template <typename T> static __device__ __forceinline__ T mmvq_s8_ld(const T * p) {
    return __ldcs(p);
}
static __device__ __forceinline__ uint32_t mmvq_s8_ld8(const uint8_t * p) {
    return mmvq_s8_ld(p);
}
static __device__ __forceinline__ uint32_t mmvq_s8_ld16(const void * p) {
    return mmvq_s8_ld((const unsigned short *) p);
}
// same as get_int_b2 (2 byte alignment)
static __device__ __forceinline__ uint32_t mmvq_s8_ld_b2(const void * x, const int i32) {
    const unsigned short * x16 = (const unsigned short *) x;
    return mmvq_s8_ld(x16 + 2*i32 + 0) | ((uint32_t) mmvq_s8_ld(x16 + 2*i32 + 1) << 16);
}
// same as get_int_b4 (4 byte alignment)
static __device__ __forceinline__ uint32_t mmvq_s8_ld_b4(const void * x, const int i32) {
    return mmvq_s8_ld((const unsigned int *) x + i32);
}

// 7 explicit sign bits + implicit 8th (even parity), see unpack_ksigns.
static __device__ __forceinline__ uint32_t mmvq_s8_ksigns(const uint32_t v7) {
    return v7 | ((__popc(v7) & 1) << 7);
}

template <ggml_type type> struct mmvq_s8_dec {
    static constexpr bool supported = false;
};

template <> struct mmvq_s8_dec<GGML_TYPE_IQ3_S> {
    static constexpr bool supported = true;
    static constexpr bool needs_ysum = false;
    struct raw { uint32_t q0, q1, qh, signs, sc, d; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_iq3_s * b = (const block_iq3_s *) vx + kbx;
        raw r;
        r.q0    = mmvq_s8_ld_b2(b->qs, 2*s + 0);
        r.q1    = mmvq_s8_ld_b2(b->qs, 2*s + 1);
        r.qh    = mmvq_s8_ld8(b->qh + s);
        r.signs = mmvq_s8_ld_b2(b->signs, s);
        r.sc    = mmvq_s8_ld8(b->scales + s/2);
        r.d     = mmvq_s8_ld16(&b->d);
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int, int * w) {
        // spread the 9th index bit of grid entry l into byte l (value 0/1)
        const uint32_t qh_lo  = ((r.qh & 0x0F) * 0x00204081u) & 0x01010101u;
        const uint32_t qh_hi  = ((r.qh >> 4)   * 0x00204081u) & 0x01010101u;
        const uint32_t s_even = r.signs & 0x0F0F0F0Fu;
        const uint32_t s_odd  = (r.signs >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const uint32_t q  = l < 4 ? r.q0 : r.q1;
            const uint32_t hb = l < 4 ? qh_lo : qh_hi;
            const uint32_t b  = l & 3;
            // 16-bit index {qh bit, qs byte} with zero upper half (sign-replicate of a byte whose msb is 0)
            const uint32_t idx = mmvq_s8_prmt(q, hb, b | ((4 + b) << 4) | ((0xC + b) << 8) | ((0xC + b) << 12));
            const uint32_t nib = mmvq_s8_prmt((l & 1) ? s_odd : s_even, 0, (l/2) | 0x4440);
            w[l] = mmvq_s8_sign4(iq3s_grid[idx], nib);
        }
    }
    static __device__ __forceinline__ float xscale(const raw & r, const int s) {
        return __half2float(__ushort_as_half((unsigned short) r.d)) * (float) (1 + 2*((r.sc >> (4*(s & 1))) & 0x0F));
    }
    static __device__ __forceinline__ int combine(const raw &, const int, const int lo, const int hi) {
        return lo + hi;
    }
    static __device__ __forceinline__ float fin(const raw & r, const int s, const int lo, const int hi, const int, const int) {
        return xscale(r, s) * (float) combine(r, s, lo, hi);
    }
    // prep/finp: the same math as fin with the per-row scale work hoisted out of the column loop (wide kernel)
    struct pre { float xs; };
    static __device__ __forceinline__ pre prep(const raw & r, const int s) {
        return { xscale(r, s) };
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int, const int) {
        return p.xs * (float) (lo + hi);
    }
};

template <> struct mmvq_s8_dec<GGML_TYPE_IQ3_XXS> {
    static constexpr bool supported = true;
    static constexpr bool needs_ysum = false;
    struct raw { uint32_t q0, q1, aux, d; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_iq3_xxs * b = (const block_iq3_xxs *) vx + kbx;
        raw r;
        r.q0  = mmvq_s8_ld_b2(b->qs, 2*s + 0);
        r.q1  = mmvq_s8_ld_b2(b->qs, 2*s + 1);
        r.aux = mmvq_s8_ld_b2(b->qs, QK_K/16 + s);
        r.d   = mmvq_s8_ld16(&b->d);
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int, int * w) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q  = k < 2 ? r.q0 : r.q1;
            const uint32_t g0 = iq3xxs_grid[mmvq_s8_prmt(q, 0, ((2*k + 0) & 3) | 0x4440)];
            const uint32_t g1 = iq3xxs_grid[mmvq_s8_prmt(q, 0, ((2*k + 1) & 3) | 0x4440)];
            const uint32_t s8 = mmvq_s8_ksigns((r.aux >> (7*k)) & 0x7F);
            w[2*k + 0] = mmvq_s8_sign4(g0, s8 & 0x0F);
            w[2*k + 1] = mmvq_s8_sign4(g1, s8 >> 4);
        }
    }
    static __device__ __forceinline__ float xscale(const raw & r, const int) {
        return __half2float(__ushort_as_half((unsigned short) r.d));
    }
    static __device__ __forceinline__ int combine(const raw & r, const int, const int lo, const int hi) {
        const int sumi = lo + hi;
        const int ls   = r.aux >> 28;
        return (ls*sumi + sumi/2)/2;
    }
    static __device__ __forceinline__ float fin(const raw & r, const int s, const int lo, const int hi, const int, const int) {
        return xscale(r, s) * (float) combine(r, s, lo, hi);
    }
    struct pre { float d; int ls; };
    static __device__ __forceinline__ pre prep(const raw & r, const int s) {
        return { xscale(r, s), (int) (r.aux >> 28) };
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int, const int) {
        const int sumi = lo + hi;
        return p.d * (float) ((p.ls*sumi + sumi/2)/2);
    }
};

template <> struct mmvq_s8_dec<GGML_TYPE_IQ2_XXS> {
    static constexpr bool supported = true;
    static constexpr bool needs_ysum = false;
    struct raw { uint32_t q, aux, d; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_iq2_xxs * b = (const block_iq2_xxs *) vx + kbx;
        raw r;
        r.q   = mmvq_s8_ld_b2(b->qs, 2*s + 0);
        r.aux = mmvq_s8_ld_b2(b->qs, 2*s + 1);
        r.d   = mmvq_s8_ld16(&b->d);
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int, int * w) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint2    g  = ((const uint2 *) iq2xxs_grid)[mmvq_s8_prmt(r.q, 0, k | 0x4440)];
            const uint32_t s8 = mmvq_s8_ksigns((r.aux >> (7*k)) & 0x7F);
            w[2*k + 0] = mmvq_s8_sign4(g.x, s8 & 0x0F);
            w[2*k + 1] = mmvq_s8_sign4(g.y, s8 >> 4);
        }
    }
    static __device__ __forceinline__ float xscale(const raw & r, const int) {
        return __half2float(__ushort_as_half((unsigned short) r.d));
    }
    static __device__ __forceinline__ int combine(const raw & r, const int, const int lo, const int hi) {
        const int ls = r.aux >> 27 | 1;
        return (lo + hi) * ls / 8;
    }
    static __device__ __forceinline__ float fin(const raw & r, const int s, const int lo, const int hi, const int, const int) {
        return xscale(r, s) * (float) combine(r, s, lo, hi);
    }
    struct pre { float d; int ls; };
    static __device__ __forceinline__ pre prep(const raw & r, const int s) {
        return { xscale(r, s), (int) (r.aux >> 27 | 1) };
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int, const int) {
        return p.d * (float) ((lo + hi) * p.ls / 8);
    }
};

template <> struct mmvq_s8_dec<GGML_TYPE_IQ2_XS> {
    static constexpr bool supported = true;
    static constexpr bool needs_ysum = false;
    struct raw { uint32_t q0, q1, sc, d; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_iq2_xs * b = (const block_iq2_xs *) vx + kbx;
        raw r;
        r.q0 = mmvq_s8_ld_b2(b->qs, 2*s + 0);
        r.q1 = mmvq_s8_ld_b2(b->qs, 2*s + 1);
        r.sc = mmvq_s8_ld8(b->scales + s);
        r.d  = mmvq_s8_ld16(&b->d);
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int, int * w) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q16 = (k < 2 ? r.q0 : r.q1) >> (16*(k & 1));
            const uint2    g   = ((const uint2 *) iq2xs_grid)[q16 & 0x1FF];
            const uint32_t s8  = mmvq_s8_ksigns((q16 >> 9) & 0x7F);
            w[2*k + 0] = mmvq_s8_sign4(g.x, s8 & 0x0F);
            w[2*k + 1] = mmvq_s8_sign4(g.y, s8 >> 4);
        }
    }
    static __device__ __forceinline__ float xscale(const raw & r, const int) {
        return __half2float(__ushort_as_half((unsigned short) r.d));
    }
    static __device__ __forceinline__ int combine(const raw & r, const int, const int lo, const int hi) {
        const int ls0 = r.sc & 0x0F;
        const int ls1 = r.sc >> 4;
        return (lo*ls0 + hi*ls1 + (lo + hi)/2)/4;
    }
    static __device__ __forceinline__ float fin(const raw & r, const int s, const int lo, const int hi, const int, const int) {
        return xscale(r, s) * (float) combine(r, s, lo, hi);
    }
    struct pre { float d; int ls0, ls1; };
    static __device__ __forceinline__ pre prep(const raw & r, const int s) {
        return { xscale(r, s), (int) (r.sc & 0x0F), (int) (r.sc >> 4) };
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int, const int) {
        return p.d * (float) ((lo*p.ls0 + hi*p.ls1 + (lo + hi)/2)/4);
    }
};

template <> struct mmvq_s8_dec<GGML_TYPE_IQ2_S> {
    static constexpr bool supported = true;
    static constexpr bool needs_ysum = false;
    struct raw { uint32_t q, signs, qh, sc, d; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_iq2_s * b = (const block_iq2_s *) vx + kbx;
        raw r;
        r.q     = mmvq_s8_ld_b2(b->qs, s);
        r.signs = mmvq_s8_ld_b2(b->qs, QK_K/32 + s);
        r.qh    = mmvq_s8_ld8(b->qh + s);
        r.sc    = mmvq_s8_ld8(b->scales + s);
        r.d     = mmvq_s8_ld16(&b->d);
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int, int * w) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t idx = mmvq_s8_prmt(r.q, 0, k | 0x4440) | ((r.qh << (8 - 2*k)) & 0x300);
            const uint2    g   = ((const uint2 *) iq2s_grid)[idx];
            const uint32_t sb  = mmvq_s8_prmt(r.signs, 0, k | 0x4440);
            w[2*k + 0] = mmvq_s8_sign4(g.x, sb & 0x0F);
            w[2*k + 1] = mmvq_s8_sign4(g.y, sb >> 4);
        }
    }
    static __device__ __forceinline__ float xscale(const raw & r, const int) {
        return __half2float(__ushort_as_half((unsigned short) r.d));
    }
    static __device__ __forceinline__ int combine(const raw & r, const int, const int lo, const int hi) {
        const int ls0 = r.sc & 0x0F;
        const int ls1 = r.sc >> 4;
        return (lo*ls0 + hi*ls1 + (lo + hi)/2)/4;
    }
    static __device__ __forceinline__ float fin(const raw & r, const int s, const int lo, const int hi, const int, const int) {
        return xscale(r, s) * (float) combine(r, s, lo, hi);
    }
    struct pre { float d; int ls0, ls1; };
    static __device__ __forceinline__ pre prep(const raw & r, const int s) {
        return { xscale(r, s), (int) (r.sc & 0x0F), (int) (r.sc >> 4) };
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int, const int) {
        return p.d * (float) ((lo*p.ls0 + hi*p.ls1 + (lo + hi)/2)/4);
    }
};

template <> struct mmvq_s8_dec<GGML_TYPE_IQ4_XS> {
    static constexpr bool supported = true;
    static constexpr bool needs_ysum = false;
    struct raw { uint32_t q[4]; uint32_t sl, sh, d; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_iq4_xs * b = (const block_iq4_xs *) vx + kbx;
        raw r;
        // block_iq4_xs is 136 bytes and qs starts at byte 8, so the 16 bytes of a slice are 8-byte aligned
        const uint2 a = mmvq_s8_ld((const uint2 *) (b->qs + 16*s) + 0);
        const uint2 c = mmvq_s8_ld((const uint2 *) (b->qs + 16*s) + 1);
        r.q[0] = a.x; r.q[1] = a.y; r.q[2] = c.x; r.q[3] = c.y;
        r.sl = mmvq_s8_ld8(b->scales_l + s/2);
        r.sh = mmvq_s8_ld16(&b->scales_h);
        r.d  = mmvq_s8_ld16(&b->d);
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int, int * w) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16(r.q[j], kvalues_iq4nl);
            w[j + 0] = v.x;
            w[j + 4] = v.y;
        }
    }
    static __device__ __forceinline__ float xscale(const raw & r, const int) {
        return __half2float(__ushort_as_half((unsigned short) r.d));
    }
    static __device__ __forceinline__ int combine(const raw & r, const int s, const int lo, const int hi) {
        const int ls = ((r.sl >> (4*(s & 1))) & 0x0F) | (((r.sh >> (2*s)) & 0x03) << 4);
        return (lo + hi) * (ls - 32);
    }
    static __device__ __forceinline__ float fin(const raw & r, const int s, const int lo, const int hi, const int, const int) {
        return xscale(r, s) * (float) combine(r, s, lo, hi);
    }
    struct pre { float d; int ls; };
    static __device__ __forceinline__ pre prep(const raw & r, const int s) {
        return { xscale(r, s), (int) (((r.sl >> (4*(s & 1))) & 0x0F) | (((r.sh >> (2*s)) & 0x03) << 4)) - 32 };
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int, const int) {
        return p.d * (float) ((lo + hi) * p.ls);
    }
};
template <> struct mmvq_s8_dec<GGML_TYPE_Q4_K> {
    static constexpr bool supported  = true;
    static constexpr bool needs_ysum = true;
    struct raw { uint32_t q[8]; uint32_t sa, sb, sc; uint32_t dm; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_q4_K * b = (const block_q4_K *) vx + kbx;
        raw r;
        // block_q4_K is 144 bytes with qs at byte 16: the 32 bytes of a slice pair are 16-byte aligned
        const int4 a = mmvq_s8_ld((const int4 *) (b->qs + 32*(s/2)) + 0);
        const int4 c = mmvq_s8_ld((const int4 *) (b->qs + 32*(s/2)) + 1);
        r.q[0] = a.x; r.q[1] = a.y; r.q[2] = a.z; r.q[3] = a.w;
        r.q[4] = c.x; r.q[5] = c.y; r.q[6] = c.z; r.q[7] = c.w;
        r.sa = mmvq_s8_ld_b4(b->scales, 0);
        r.sb = mmvq_s8_ld_b4(b->scales, 1);
        r.sc = mmvq_s8_ld_b4(b->scales, 2);
        r.dm = mmvq_s8_ld((const unsigned int *) &b->dm);
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int s, int * w) {
        const int shift = 4*(s & 1);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            w[l] = (r.q[l] >> shift) & 0x0F0F0F0F;
        }
    }
    static __device__ __forceinline__ float fin(const raw & r, const int s, const int lo, const int hi, const int su_lo, const int su_hi) {
        // get_scale_min_k4, branch-free on the lane-dependent sub-block index s
        const uint32_t sel = (s & 3) | 0x4440;
        const uint32_t a = mmvq_s8_prmt(r.sa, 0, sel);
        const uint32_t b = mmvq_s8_prmt(r.sb, 0, sel);
        const uint32_t c = mmvq_s8_prmt(r.sc, 0, sel);
        const int sc = s < 4 ? (a & 63) : ((c & 0x0F) | ((a >> 2) & 0x30));
        const int m  = s < 4 ? (b & 63) : ((c >> 4)   | ((b >> 2) & 0x30));
        const float2 dm = __half22float2(*(const half2 *) &r.dm);
        return dm.x*(float) (sc*(lo + hi)) - dm.y*(float) (m*(su_lo + su_hi));
    }
    struct pre { int sc, m; float2 dm; };
    static __device__ __forceinline__ pre prep(const raw & r, const int s) {
        const uint32_t sel = (s & 3) | 0x4440;
        const uint32_t a = mmvq_s8_prmt(r.sa, 0, sel);
        const uint32_t b = mmvq_s8_prmt(r.sb, 0, sel);
        const uint32_t c = mmvq_s8_prmt(r.sc, 0, sel);
        pre p;
        p.sc = s < 4 ? (a & 63) : ((c & 0x0F) | ((a >> 2) & 0x30));
        p.m  = s < 4 ? (b & 63) : ((c >> 4)   | ((b >> 2) & 0x30));
        p.dm = __half22float2(*(const half2 *) &r.dm);
        return p;
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int su_lo, const int su_hi) {
        return p.dm.x*(float) (p.sc*(lo + hi)) - p.dm.y*(float) (p.m*(su_lo + su_hi));
    }
};

// IQ1_M: grid values g in {0, 1, 2} encode the weights g - 1 + delta with delta = +-IQ1M_DELTA = +-1/8, so the decoded
// int8 weights are 8*g + 1 + delta*8 in [0, 18] and fin removes the offset with the activation sums:
// 8 * sum((g - 1 + delta)*y) = sum(w*y) - 9*sum(y). Same result as vec_dot_iq1_m_q8_1 (both are exact before the scale).
template <> struct mmvq_s8_dec<GGML_TYPE_IQ1_M> {
    static constexpr bool supported  = true;
    static constexpr bool needs_ysum = true;
    struct raw { uint32_t qs, qh, sc01, sc23; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_iq1_m * b = (const block_iq1_m *) vx + kbx;
        raw r;
        // block_iq1_m is 56 bytes: qs (4-byte aligned), qh, then the 8 scale bytes (8-byte aligned)
        r.qs = mmvq_s8_ld_b4(b->qs, s);
        r.qh = mmvq_s8_ld16(b->qh + 2*s);
        const uint2 sc = mmvq_s8_ld((const uint2 *) b->scales);
        r.sc01 = sc.x;
        r.sc23 = sc.y;
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int, int * w) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t qhl  = r.qh >> (4*k);
            const uint32_t grid = iq1s_grid_gpu[mmvq_s8_prmt(r.qs, 0, k | 0x4440) | ((qhl & 0x07) << 8)];
            const uint32_t sb   = (qhl & 0x08) ? 0u : 0x02020202u; // 1 + 8*delta
            w[2*k + 0] = (int) ((((grid >> 0) & 0x0F0F0F0F) << 3) | sb);
            w[2*k + 1] = (int) ((((grid >> 4) & 0x0F0F0F0F) << 3) | sb);
        }
    }
    struct pre { float d; int sc0, sc1; };
    static __device__ __forceinline__ pre prep(const raw & r, const int s) {
        iq1m_scale_t scale;
        scale.u16 = ((r.sc01 & 0xFFFF) >> 12) | ((r.sc01 >> 24) & 0x00F0) | ((r.sc23 >> 4) & 0x0F00) | ((r.sc23 >> 16) & 0xF000);
        const uint32_t sc  = (s < 4 ? r.sc01 : r.sc23) >> (16*((s/2) & 1));
        const int      tmp = (sc & 0xFFFF) >> (6*(s & 1));
        return { __half2float(scale.f16) * (1.0f/8.0f), 2*((tmp >> 0) & 0x07) + 1, 2*((tmp >> 3) & 0x07) + 1 };
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int su_lo, const int su_hi) {
        return p.d * (float) ((lo - 9*su_lo)*p.sc0 + (hi - 9*su_hi)*p.sc1);
    }
    static __device__ __forceinline__ float fin(const raw & r, const int s, const int lo, const int hi, const int su_lo, const int su_hi) {
        return finp(prep(r, s), lo, hi, su_lo, su_hi);
    }
};

template <> struct mmvq_s8_dec<GGML_TYPE_Q2_K> {
    static constexpr bool supported  = true;
    static constexpr bool needs_ysum = true;
    struct raw { uint32_t q[8]; uint32_t sc; uint32_t dm; };
    static __device__ __forceinline__ raw load(const void * GGML_CUDA_RESTRICT vx, const int kbx, const int s) {
        const block_q2_K * b = (const block_q2_K *) vx + kbx;
        raw r;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            r.q[l] = mmvq_s8_ld_b4(b->qs, 8*(s/4) + l);
        }
        r.sc = mmvq_s8_ld16(b->scales + 2*s);
        r.dm = mmvq_s8_ld((const unsigned int *) &b->dm);
        return r;
    }
    static __device__ __forceinline__ void decode(const raw & r, const int s, int * w) {
        const int shift = 2*(s & 3);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            w[l] = (r.q[l] >> shift) & 0x03030303;
        }
    }
    static __device__ __forceinline__ float fin(const raw & r, const int, const int lo, const int hi, const int su_lo, const int su_hi) {
        const int sc_lo = r.sc & 0xFF;
        const int sc_hi = r.sc >> 8;
        const float2 dm = __half22float2(*(const half2 *) &r.dm);
        return dm.x*(float) ((sc_lo & 0x0F)*lo + (sc_hi & 0x0F)*hi) - dm.y*(float) ((sc_lo >> 4)*su_lo + (sc_hi >> 4)*su_hi);
    }
    struct pre { int sc_lo, sc_hi; float2 dm; };
    static __device__ __forceinline__ pre prep(const raw & r, const int) {
        return { (int) (r.sc & 0xFF), (int) (r.sc >> 8), __half22float2(*(const half2 *) &r.dm) };
    }
    static __device__ __forceinline__ float finp(const pre & p, const int lo, const int hi, const int su_lo, const int su_hi) {
        return p.dm.x*(float) ((p.sc_lo & 0x0F)*lo + (p.sc_hi & 0x0F)*hi) - p.dm.y*(float) ((p.sc_lo >> 4)*su_lo + (p.sc_hi >> 4)*su_hi);
    }
};

// One warp per block computes rows_per_block rows for all ncols_dst columns: per iteration it covers 4 super blocks,
// 8 lanes each. The raw weight bits of the next iteration are loaded right after a row has been decoded so they are in
// flight during the remaining rows' math, activations are loaded once per iteration and reused for all rows, and each
// weight is decoded once for all columns. Rows past nrows_x are clamped for loading and not written.
// has_gate: fused gate/up(+GLU) pair, the gate weights may use another quant type (type_gate). Biases (fusion.x_bias,
// fusion.gate_bias) are added at runtime with their own column strides and the channel/sample strides of dst.
template <ggml_type type, int ncols_dst, bool has_gate, int rows_per_block, ggml_type type_gate = type>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 12) // <= 170 registers -> at least 12 resident warps per SM
static __global__ void mul_mat_vec_q_s8(
        const void * vx_ptr, const void * vy_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr, const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst, const uint32_t stride_col_x_bias,
        const uint32_t stride_col_gate_bias, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst, const bool w_static) {
    // PDL and __restrict__ must not be combined (see GGML_CUDA_RESTRICT): restrict-qualified locals, as in mul_mat_vec_q
    const void * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void * GGML_CUDA_RESTRICT vy  = vy_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;
    using dec   = mmvq_s8_dec<type>;
    using dec_g = mmvq_s8_dec<type_gate>;
    static_assert(dec::supported && dec_g::supported, "type not supported by mul_mat_vec_q_s8");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == 32, "mul_mat_vec_q_s8 assumes 32 lanes");

    const int lane = threadIdx.x;
    const int s    = lane % 8;
    const int row0 = rows_per_block*blockIdx.x;
    const int nkb  = ncols_x / QK_K;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    [[maybe_unused]] const void * vgate = fusion.gate;

    // PDL: early trigger and weight prefetch before the dependency wait, see mul_mat_vec_q_s8w
    ggml_cuda_pdl_lc();
    if (!w_static) {
        ggml_cuda_pdl_sync();
    }

    // 32-bit signed index math keeps the address arithmetic cheap (all offsets are in blocks and fit in int)
    const int x_base = sample_x*stride_sample_x + channel_x*stride_channel_x;
    int xoff[rows_per_block];
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
        xoff[i] = x_base + min(row0 + i, (int) nrows_x - 1)*(int) stride_row_x;
    }
    const int col_y = stride_col_y;
    const block_q8_1 * y = (const block_q8_1 *) vy + (int) (sample_dst*stride_sample_y + channel_dst*stride_channel_y) + s;

    float tmp[ncols_dst][rows_per_block] = {{0.0f}};
    [[maybe_unused]] float tmp_gate[ncols_dst][rows_per_block] = {{0.0f}};

    typename dec::raw cur[rows_per_block];
    [[maybe_unused]] typename dec_g::raw cur_gate[rows_per_block];

    int kbx = lane / 8;
    if (nkb > 0) {
        const int kb0 = min(kbx, nkb - 1);
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            cur[i] = dec::load(vx, xoff[i] + kb0, s);
            if constexpr (has_gate) {
                cur_gate[i] = dec_g::load(vgate, xoff[i] + kb0, s);
            }
        }
    }
    if (w_static) {
        ggml_cuda_pdl_sync();
    }

    for (; kbx < nkb; kbx += 4) {
        int   u[ncols_dst][8];
        float d8[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const block_q8_1 * yb = y + (j*col_y + kbx*(QK_K/QK8_1));
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                u[j][l] = get_int_b4(yb->qs, l);
            }
            d8[j] = __low2float(yb->ds);
        }
        [[maybe_unused]] int su_lo[ncols_dst] = {};
        [[maybe_unused]] int su_hi[ncols_dst] = {};
        if constexpr (dec::needs_ysum || (has_gate && dec_g::needs_ysum)) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                su_lo[j] = 0;
                su_hi[j] = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    su_lo[j] = ggml_cuda_dp4a(0x01010101, u[j][l + 0], su_lo[j]);
                    su_hi[j] = ggml_cuda_dp4a(0x01010101, u[j][l + 4], su_hi[j]);
                }
            }
        }

        const int kbn = min(kbx + 4, nkb - 1);
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            {
                int w[8];
                dec::decode(cur[i], s, w);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    int lo = 0;
                    int hi = 0;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        lo = ggml_cuda_dp4a(w[l + 0], u[j][l + 0], lo);
                        hi = ggml_cuda_dp4a(w[l + 4], u[j][l + 4], hi);
                    }
                    tmp[j][i] += d8[j] * dec::fin(cur[i], s, lo, hi, su_lo[j], su_hi[j]);
                }
                cur[i] = dec::load(vx, xoff[i] + kbn, s);
            }
            if constexpr (has_gate) {
                int w[8];
                dec_g::decode(cur_gate[i], s, w);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    int lo = 0;
                    int hi = 0;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        lo = ggml_cuda_dp4a(w[l + 0], u[j][l + 0], lo);
                        hi = ggml_cuda_dp4a(w[l + 4], u[j][l + 4], hi);
                    }
                    tmp_gate[j][i] += d8[j] * dec_g::fin(cur_gate[i], s, lo, hi, su_lo[j], su_hi[j]);
                }
                cur_gate[i] = dec_g::load(vgate, xoff[i] + kbn, s);
            }
        }
    }

    const int off_dst = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
    const float * x_bias    = (const float *) fusion.x_bias;
    const float * gate_bias = (const float *) fusion.gate_bias;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            float result = warp_reduce_sum<warp_size>(tmp[j][i]);
            [[maybe_unused]] float gate_value = 0.0f;
            if constexpr (has_gate) {
                gate_value = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
            }
            if (lane == i && row0 + i < (int) nrows_x) {
                if (x_bias) {
                    result += x_bias[off_dst + j*stride_col_x_bias + i];
                }
                if constexpr (has_gate) {
                    if (gate_bias) {
                        gate_value += gate_bias[off_dst + j*stride_col_gate_bias + i];
                    }
                    switch (fusion.glu_op) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        case GGML_GLU_OP_SWIGLU_CLAMP:
                            result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, fusion.glu_limit);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
                dst[off_dst + j*stride_col_dst + i] = result;
            }
        }
    }
}

// Sums the N values v[0..N) (N a power of 2, <= 32) over the 32 lanes of a warp as a reduce-scatter: every step sends
// the half of the values the lane does not keep, so the whole reduction takes N - 1 + 5 - log2(N) shuffles instead of
// 5*N. Afterwards lane l holds the total of value l >> (5 - log2(N)) in v[0].
template <int N>
static __device__ __forceinline__ float mmvq_s8_reduce_scatter(float * v, const int lane) {
    static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "N must be a power of 2 <= 32");
#pragma unroll
    for (int step = 0; step < 5; ++step) {
        const int o = 16 >> step;
        const int h = N >> (step + 1); // values kept after this step
        if (h >= 1) {
            const bool upper = lane & o;
#pragma unroll
            for (int k = 0; k < h; ++k) {
                const float send = upper ? v[k] : v[k + h];
                const float keep = upper ? v[k + h] : v[k];
                v[k] = keep + __shfl_xor_sync(0xFFFFFFFF, send, o, 32);
            }
        } else {
            v[0] += __shfl_xor_sync(0xFFFFFFFF, v[0], o, 32);
        }
    }
    return v[0];
}

static constexpr __host__ __device__ int mmvq_s8_pow2_ceil(const int n) {
    return n <= 1 ? 1 : 2*mmvq_s8_pow2_ceil((n + 1)/2);
}

// Loads the 8 quant ints and the scale d of the q8_1 block at yb (4-byte aligned, 36 bytes) with five 8-byte loads
// of the 8-byte aligned 40-byte window around it instead of nine 4-byte loads (the L1 load throughput is what limits
// the kernels below at several columns). The window reads the 4 bytes before an odd-aligned block or after an even
// one: callers only use blocks with an in-bounds neighbor on that side (see mul_mat_vec_q_s8w).
static __device__ __forceinline__ float mmvq_s8_load_q8_1(const block_q8_1 * yb, int * u) {
    const uint2 * w8  = (const uint2 *) ((uintptr_t) yb & ~(uintptr_t) 7);
    const bool    odd = ((uintptr_t) yb) & 4;
    uint32_t w[10];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        const uint2 t = w8[q];
        w[2*q + 0] = t.x;
        w[2*q + 1] = t.y;
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        u[l] = odd ? w[l + 2] : w[l + 1];
    }
    const uint32_t ds = odd ? w[1] : w[0];
    return __low2float(*(const half2 *) &ds);
}

// Columns-outer variant of mul_mat_vec_q_s8 ("dequantize once, dot many"): per iteration every row's weight slice is
// decoded first (all rows' raw bits of the next iteration are then in flight during the whole column loop), then the
// activations are loaded one column at a time and dotted with all rows, so the registers scale with rows + columns
// instead of rows * columns; the per-row scale work (prep) is hoisted out of the column loop and the final warp
// reduction is a reduce-scatter.
// ncols_max: compile-time column count; the runtime ncols_dst may be smaller (one instance serves several column
// counts): the missing columns repeat the last column (loads stay in bounds, no branches in the column loop) and are
// not written. ksplit > 1: the K range is split over the ksplit warps of a block (combined via shared memory), for
// matrices with too few rows to fill the GPU. minb: resident blocks per SM the registers are budgeted for.
// pipe: the activations of the next column (and of the first column of the next iteration) are loaded before the
// current column's math (measured faster for up to 4 columns, neutral to slower above).
// The q8_1 blocks of every column must start 8-byte aligned at even block indices (checked by the host code): lane s
// reads the 4 bytes after block 8*kbx + s for even s and before it for odd s, i.e. always inside the same column.
template <ggml_type type, int ncols_max, bool has_gate, int rows_per_block, int ksplit, ggml_type type_gate = type,
          int minb = 12, bool pipe = false>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*ksplit, minb/ksplit)
static __global__ void mul_mat_vec_q_s8w(
        const void * vx_ptr, const void * vy_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr, const uint32_t ncols_x, const uint32_t nrows_x, const int ncols_dst,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst, const uint32_t stride_col_x_bias,
        const uint32_t stride_col_gate_bias, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst, const bool w_static) {
    // PDL and __restrict__ must not be combined (see GGML_CUDA_RESTRICT): restrict-qualified locals, as in mul_mat_vec_q
    const void * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void * GGML_CUDA_RESTRICT vy  = vy_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;
    using dec   = mmvq_s8_dec<type>;
    using dec_g = mmvq_s8_dec<type_gate>;
    static_assert(dec::supported && dec_g::supported, "type not supported by mul_mat_vec_q_s8w");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == 32, "mul_mat_vec_q_s8w assumes 32 lanes");
    constexpr int ng   = has_gate ? 2 : 1;
    constexpr int nval = ncols_max*rows_per_block*ng;
    constexpr int npad = mmvq_s8_pow2_ceil(nval);
    constexpr int nch  = npad > 32 ? npad/32 : 1; // reduce-scatter chunks of up to 32 values
    constexpr int ncv  = npad/nch;                 // values per chunk
    constexpr bool ysum = dec::needs_ysum || (has_gate && dec_g::needs_ysum);

    const int lane = threadIdx.x;
    const int wk   = ksplit > 1 ? (int) threadIdx.y : 0;
    const int s    = lane % 8;
    const int row0 = rows_per_block*blockIdx.x;
    const int nkb  = ncols_x / QK_K;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    [[maybe_unused]] const void * vgate = fusion.gate;

    // programmatic dependent launch: the next kernel may start as soon as every block of this one runs (its blocks fill
    // the slots of the finished ones and prefetch their weights while this kernel's last blocks finish, measured +2..10%
    // for few-row products), it waits for this kernel's completion before reading anything this kernel writes.
    // w_static: the weights are not written by the preceding kernels (model weights), so the first iteration's weight
    // loads are issued before waiting for them, everything else is read afterwards.
    ggml_cuda_pdl_lc();
    if (!w_static) {
        ggml_cuda_pdl_sync();
    }

    const int x_base = sample_x*stride_sample_x + channel_x*stride_channel_x;
    int xoff[rows_per_block];
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
        xoff[i] = x_base + min(row0 + i, (int) nrows_x - 1)*(int) stride_row_x;
    }
    const int col_y = stride_col_y;
    const block_q8_1 * y = (const block_q8_1 *) vy + (int) (sample_dst*stride_sample_y + channel_dst*stride_channel_y) + s;

    // this warp's K range: iterations [it0, it1) of 4 super blocks
    const int niter  = (nkb + 3) / 4;
    const int it0    = (wk*niter) / ksplit;
    const int it1    = ((wk + 1)*niter) / ksplit;
    const int kb_end = min(nkb, 4*it1);

    // acc[(j*rows_per_block + i)*ng + g]: column j, row i, g = 1: gate
    float acc[npad];
#pragma unroll
    for (int v = 0; v < npad; ++v) {
        acc[v] = 0.0f;
    }

    typename dec::raw cur[rows_per_block];
    [[maybe_unused]] typename dec_g::raw cur_gate[rows_per_block];

    int kbx = 4*it0 + lane / 8;
    if (kb_end > 4*it0) {
        const int kb0 = min(kbx, kb_end - 1);
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            cur[i] = dec::load(vx, xoff[i] + kb0, s);
            if constexpr (has_gate) {
                cur_gate[i] = dec_g::load(vgate, xoff[i] + kb0, s);
            }
        }
    }
    if (w_static) {
        ggml_cuda_pdl_sync();
    }

    // pipe: activations of the next column to be used, loaded one column ahead
    [[maybe_unused]] int   u_next[8];
    [[maybe_unused]] float d_next = 0.0f;
    if constexpr (pipe) {
        d_next = mmvq_s8_load_q8_1(y + min(kbx, max(kb_end - 1, 0))*(QK_K/QK8_1), u_next);
    }

    for (; kbx < kb_end; kbx += 4) {
        const int kbn = min(kbx + 4, kb_end - 1);

        int w[rows_per_block][8];
        typename dec::pre p[rows_per_block];
        [[maybe_unused]] int wg[rows_per_block][8];
        [[maybe_unused]] typename dec_g::pre pg[rows_per_block];
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            dec::decode(cur[i], s, w[i]);
            p[i]   = dec::prep(cur[i], s);
            cur[i] = dec::load(vx, xoff[i] + kbn, s);
            if constexpr (has_gate) {
                dec_g::decode(cur_gate[i], s, wg[i]);
                pg[i]       = dec_g::prep(cur_gate[i], s);
                cur_gate[i] = dec_g::load(vgate, xoff[i] + kbn, s);
            }
        }

        const block_q8_1 * yk = y + kbx*(QK_K/QK8_1);
#pragma unroll
        for (int j = 0; j < ncols_max; ++j) {
            int   u[8];
            float d8;
            if constexpr (pipe) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    u[l] = u_next[l];
                }
                d8 = d_next;
                if (j + 1 < ncols_max) {
                    d_next = mmvq_s8_load_q8_1(yk + min(j + 1, ncols_dst - 1)*col_y, u_next);
                } else {
                    d_next = mmvq_s8_load_q8_1(y + kbn*(QK_K/QK8_1), u_next); // first column of the next iteration
                }
            } else {
                d8 = mmvq_s8_load_q8_1(yk + min(j, ncols_dst - 1)*col_y, u);
            }
            [[maybe_unused]] int su_lo = 0;
            [[maybe_unused]] int su_hi = 0;
            if constexpr (ysum) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    su_lo = ggml_cuda_dp4a(0x01010101, u[l + 0], su_lo);
                    su_hi = ggml_cuda_dp4a(0x01010101, u[l + 4], su_hi);
                }
            }
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                int lo = 0;
                int hi = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    lo = ggml_cuda_dp4a(w[i][l + 0], u[l + 0], lo);
                    hi = ggml_cuda_dp4a(w[i][l + 4], u[l + 4], hi);
                }
                acc[(j*rows_per_block + i)*ng] += d8 * dec::finp(p[i], lo, hi, su_lo, su_hi);
                if constexpr (has_gate) {
                    int glo = 0;
                    int ghi = 0;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        glo = ggml_cuda_dp4a(wg[i][l + 0], u[l + 0], glo);
                        ghi = ggml_cuda_dp4a(wg[i][l + 4], u[l + 4], ghi);
                    }
                    acc[(j*rows_per_block + i)*ng + 1] += d8 * dec_g::finp(pg[i], glo, ghi, su_lo, su_hi);
                }
            }
        }
    }

    constexpr int shift = 5 - (ncv == 1 ? 0 : ncv == 2 ? 1 : ncv == 4 ? 2 : ncv == 8 ? 3 : ncv == 16 ? 4 : 5);
    float res[nch];
#pragma unroll
    for (int c = 0; c < nch; ++c) {
        res[c] = mmvq_s8_reduce_scatter<ncv>(acc + c*ncv, lane);
    }
    if constexpr (ksplit > 1) {
        __shared__ float red[ksplit > 1 ? ksplit - 1 : 1][nch][32];
        if (wk > 0) {
#pragma unroll
            for (int c = 0; c < nch; ++c) {
                red[wk - 1][c][lane] = res[c];
            }
        }
        __syncthreads();
        if (wk > 0) {
            return;
        }
#pragma unroll
        for (int k = 0; k < ksplit - 1; ++k) {
#pragma unroll
            for (int c = 0; c < nch; ++c) {
                res[c] += red[k][c][lane];
            }
        }
    }
    const int off_dst = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
    const float * x_bias    = (const float *) fusion.x_bias;
    const float * gate_bias = (const float *) fusion.gate_bias;
#pragma unroll
    for (int c = 0; c < nch; ++c) {
        float result = res[c];
        const int idx = c*ncv + (lane >> shift);
        [[maybe_unused]] float gate_value = 0.0f;
        if constexpr (has_gate) {
            // the gate total of value idx is value idx + 1, i.e. in lane + (1 << shift)
            gate_value = __shfl_xor_sync(0xFFFFFFFF, result, 1 << shift, 32);
        }
        const int j = idx / (rows_per_block*ng);
        const int i = (idx / ng) % rows_per_block;
        const bool writer = (lane & ((1 << shift) - 1)) == 0 && (!has_gate || idx % 2 == 0);
        if (writer && idx < nval && j < ncols_dst && row0 + i < (int) nrows_x) {
            if (x_bias) {
                result += x_bias[off_dst + j*stride_col_x_bias + i];
            }
            if constexpr (has_gate) {
                if (gate_bias) {
                    gate_value += gate_bias[off_dst + j*stride_col_gate_bias + i];
                }
                switch (fusion.glu_op) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, fusion.glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
            dst[off_dst + j*stride_col_dst + i] = result;
        }
    }
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Extra arguments of the few-column kernel: the gate weight type of a fused gate/up pair (it may differ from src0's,
// see ggml_cuda_mmvq_s8_can_fuse_mixed; GGML_TYPE_COUNT = same type) and the column strides of the fused biases.
struct mmvq_s8_args {
    ggml_type type_gate            = GGML_TYPE_COUNT;
    int       stride_col_x_bias    = 0;
    int       stride_col_gate_bias = 0;
    bool      w_static             = false; // weights (and gate) are model weights, not written by preceding kernels
};

// Weights in a model weight buffer are never written by graph kernels: the few-column kernels may then load them before
// waiting for the preceding kernel (programmatic dependent launch).
static bool ggml_cuda_mmvq_s8_weights_static(const ggml_tensor * w) {
    return w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static bool ggml_cuda_mmvq_s8_type(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_mmvq_s8_usable(const ggml_type type, const int cc, const int64_t ncols_dst, const bool has_ids) {
#ifdef GGML_CUDA_MMVQ_S8_AVAILABLE
    // Tuned and tested on RTX 5070 Ti (sm_120) only; other GPUs keep the generic kernel.
    if (has_ids || ncols_dst < 2 || ncols_dst > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_BLACKWELL || cc >= GGML_CUDA_CC_DGX_SPARK) {
        return false;
    }
    return ggml_cuda_mmvq_s8_type(type);
#else
    GGML_UNUSED_VARS(type, cc, ncols_dst, has_ids);
    return false;
#endif // GGML_CUDA_MMVQ_S8_AVAILABLE
}

// Mixed-type gate/up fusion (e.g. IQ3_XXS gate + IQ3_S up) for the speculative-verify batches of 2..8 columns, for plain
// 2D contiguous weights with the same number of 256-weight super blocks per row.
bool ggml_cuda_mmvq_s8_can_fuse_mixed(const ggml_tensor * up_w, const ggml_tensor * gate_w, const int64_t ncols_dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return up_w->type != gate_w->type &&
        ggml_cuda_mmvq_s8_usable(up_w->type, cc, ncols_dst, false) && ggml_cuda_mmvq_s8_usable(gate_w->type, cc, ncols_dst, false) &&
        ggml_are_same_shape(up_w, gate_w) && ggml_is_contiguous(up_w) && ggml_is_contiguous(gate_w) &&
        up_w->ne[2] == 1 && up_w->ne[3] == 1 && up_w->ne[0] % QK_K == 0;
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const uint32_t row = min(uint32_t(i), nrows_x - 1 - uint32_t(row0));
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + row*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + row*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false,
        const bool split_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters, split_k);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, bool split_k = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, split_k>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, split_k>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

template <int v> using mmvq_s8_ic = std::integral_constant<int, v>;

// Configurations of the few-column kernels without a fused gate, chosen per type from same-process A/B runs of the
// model's shapes on the RTX 5070 Ti (sm_120); rows = rows of dst over all channels/samples.
enum mmvq_s8_cfg {
    MMVQ_S8_RO3, // mul_mat_vec_q_s8: rows-outer loop, 3 rows per warp (2..4 columns)
    MMVQ_S8_RO4, // mul_mat_vec_q_s8: rows-outer loop, 4 rows per warp (4 columns, v2's configuration)
    MMVQ_S8_P4,  // mul_mat_vec_q_s8w: 4 rows per warp, pipelined activation loads (2..4 columns)
    MMVQ_S8_W4,  // mul_mat_vec_q_s8w: 4 rows per warp
    MMVQ_S8_K2,  // mul_mat_vec_q_s8w: 4 rows per warp, K split over the 2 warps of a block (few rows)
    MMVQ_S8_W14, // mul_mat_vec_q_s8w: 4 rows per warp, registers budgeted for 14 warps per SM (5..8 columns)
};

static mmvq_s8_cfg mmvq_s8_select(const ggml_type type, const int ncols, const int64_t rows) {
    const bool few = rows <= 5120; // <= ~1.5 waves of 4-row warps
    if (rows <= 2048) {
        return MMVQ_S8_K2;
    }
    if (ncols >= 5) {
        if (type == GGML_TYPE_Q4_K && !few) {
            return MMVQ_S8_K2;
        }
        return few ? MMVQ_S8_W4 : MMVQ_S8_W14;
    }
    switch (type) {
        case GGML_TYPE_IQ3_S:   return few ? MMVQ_S8_K2 : ncols == 2 ? MMVQ_S8_RO3 : ncols == 3 ? MMVQ_S8_W4 : MMVQ_S8_P4;
        case GGML_TYPE_IQ3_XXS: return few ? MMVQ_S8_K2 : MMVQ_S8_P4;
        case GGML_TYPE_IQ4_XS:  return few ? MMVQ_S8_P4 : ncols <= 3 ? MMVQ_S8_RO3 : MMVQ_S8_RO4;
        case GGML_TYPE_Q4_K:    return few ? MMVQ_S8_P4 : MMVQ_S8_K2;
        case GGML_TYPE_Q2_K:    return ncols <= 3 ? MMVQ_S8_RO3 : MMVQ_S8_W4;
        case GGML_TYPE_IQ2_S:   return ncols == 2 ? MMVQ_S8_RO3 : few ? MMVQ_S8_K2 : ncols == 3 ? MMVQ_S8_W4 : MMVQ_S8_P4;
        case GGML_TYPE_IQ2_XS:  return ncols == 2 ? MMVQ_S8_RO3 : ncols == 3 ? MMVQ_S8_W4 : MMVQ_S8_P4;
        case GGML_TYPE_IQ2_XXS: return ncols == 4 ? MMVQ_S8_W4 : MMVQ_S8_P4;
        default:                return MMVQ_S8_P4; // IQ1_M
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const mmvq_s8_args & s8 = {}) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    // Prefer split K for small or partial row tiles; one-warp blocks otherwise process four rows densely.
    const bool split_k = nrows_x % 4 != 0 || nrows_x < 32*ggml_cuda_info().devices[device].nsm;

    const auto launch_few_cols = [&](auto ncols_dst_tag) {
        constexpr int c_ncols_dst = decltype(ncols_dst_tag)::value;
        const auto launch = [&](auto split_k_tag) {
            constexpr bool c_split_k = decltype(split_k_tag)::value;
            const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst,
                                                                          warp_size, table_id, false, false, c_split_k);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, c_split_k>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride, stream);
        };
        if (split_k) {
            launch(std::true_type{});
        } else {
            launch(std::false_type{});
        }
    };

#ifdef GGML_CUDA_MMVQ_S8_AVAILABLE
    if constexpr (mmvq_s8_dec<type>::supported) {
        const bool has_gate = fusion.gate != nullptr;
        if (ggml_cuda_mmvq_s8_usable(type, cc, ncols_dst, has_ids) && warp_size == 32) {
            const ggml_type type_gate = s8.type_gate == GGML_TYPE_COUNT ? type : s8.type_gate;
            GGML_ASSERT(type_gate == type || has_gate);
            // mul_mat_vec_q_s8w reads q8_1 blocks in 8-byte aligned windows (mmvq_s8_load_q8_1): the q8_1 data of
            // every column must start 8-byte aligned, true for the pool buffers and q8_1 cache slots (>= 128-byte
            // aligned) with rows of GGML_PAD(ne10, MATRIX_ROW_PADDING)/QK8_1 blocks
            GGML_ASSERT((uintptr_t) vy % 8 == 0 && (ncols_dst == 1 || stride_col_y % 2 == 0) &&
                        (nchannels_dst == 1 || stride_channel_y % 2 == 0) && (nsamples_dst == 1 || stride_sample_y % 2 == 0));

            // Kernel configurations (mul_mat_vec_q_s8w), chosen from same-process A/B runs of every model shape on the
            // RTX 5070 Ti (sm_120): rows per warp R, K split over ksplit warps of a block, registers budgeted for minb
            // resident blocks per SM, pipelined activation loads.
            const auto launch_w = [&](auto ncols_tag, auto gate_tag, auto type_gate_tag, auto rows_tag, auto ksplit_tag,
                                      auto minb_tag, auto pipe_tag) {
                constexpr int       c_ncols_max    = decltype(ncols_tag)::value;
                constexpr bool      c_has_gate     = decltype(gate_tag)::value;
                constexpr ggml_type c_type_gate    = decltype(type_gate_tag)::value;
                constexpr int       rows_per_block = decltype(rows_tag)::value;
                constexpr int       c_ksplit       = decltype(ksplit_tag)::value;
                constexpr int       c_minb         = decltype(minb_tag)::value;
                constexpr bool      c_pipe         = decltype(pipe_tag)::value;
                const dim3 block_nums((nrows_x + rows_per_block - 1)/rows_per_block, nchannels_dst, nsamples_dst);
                const dim3 block_dims(warp_size, c_ksplit, 1);
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
                ggml_cuda_kernel_launch(
                    mul_mat_vec_q_s8w<type, c_ncols_max, c_has_gate, rows_per_block, c_ksplit, c_type_gate, c_minb, c_pipe>, launch_params,
                    vx, vy, fusion, dst, (uint32_t) ncols_x, (uint32_t) nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                    s8.stride_col_x_bias, s8.stride_col_gate_bias, channel_ratio_fd, stride_channel_x, stride_channel_y,
                    stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, s8.w_static);
            };
            using f_t = std::false_type;
            using t_t = std::true_type;
            // fused gate/up: 3 rows of both matrices per warp (2 for 2..3 columns and for Q4_K, measured slower with 3)
            if (type_gate != type) {
                // gate/up weights of different types: one instance per 4 / 6 / 8 columns (a smaller column count repeats
                // its last column, see mul_mat_vec_q_s8w), fewer instances for the 9x8 type pairs
                const auto launch_mixed = [&](auto type_gate_tag) {
                    constexpr bool q4k = type == GGML_TYPE_Q4_K || decltype(type_gate_tag)::value == GGML_TYPE_Q4_K;
                    using rows = mmvq_s8_ic<q4k ? 2 : 3>;
                    if (ncols_dst <= 4) {
                        launch_w(mmvq_s8_ic<4>{}, t_t{}, type_gate_tag, rows{}, mmvq_s8_ic<1>{}, mmvq_s8_ic<12>{}, f_t{});
                    } else if (ncols_dst <= 6) {
                        launch_w(mmvq_s8_ic<6>{}, t_t{}, type_gate_tag, rows{}, mmvq_s8_ic<1>{}, mmvq_s8_ic<12>{}, f_t{});
                    } else {
                        launch_w(mmvq_s8_ic<8>{}, t_t{}, type_gate_tag, rows{}, mmvq_s8_ic<1>{}, mmvq_s8_ic<12>{}, f_t{});
                    }
                };
                switch (type_gate) {
                    case GGML_TYPE_IQ3_S:   launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_IQ3_S>{});   break;
                    case GGML_TYPE_IQ3_XXS: launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_IQ3_XXS>{}); break;
                    case GGML_TYPE_IQ2_XXS: launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_IQ2_XXS>{}); break;
                    case GGML_TYPE_IQ2_XS:  launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_IQ2_XS>{});  break;
                    case GGML_TYPE_IQ2_S:   launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_IQ2_S>{});   break;
                    case GGML_TYPE_IQ4_XS:  launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_IQ4_XS>{});  break;
                    case GGML_TYPE_Q4_K:    launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{});    break;
                    case GGML_TYPE_Q2_K:    launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_Q2_K>{});    break;
                    case GGML_TYPE_IQ1_M:   launch_mixed(std::integral_constant<ggml_type, GGML_TYPE_IQ1_M>{});   break;
                    default: GGML_ABORT("fatal error");
                }
                return;
            }
            using same_t = std::integral_constant<ggml_type, type>;
            if (has_gate) {
                const auto launch_g = [&](auto ncols_tag) {
                    constexpr int c_ncols = decltype(ncols_tag)::value;
                    using rows = mmvq_s8_ic<c_ncols <= 3 || type == GGML_TYPE_Q4_K ? 2 : 3>;
                    launch_w(ncols_tag, t_t{}, same_t{}, rows{}, mmvq_s8_ic<1>{}, mmvq_s8_ic<12>{}, f_t{});
                };
                switch (ncols_dst) {
                    case 2: launch_g(mmvq_s8_ic<2>{}); break;
                    case 3: launch_g(mmvq_s8_ic<3>{}); break;
                    case 4: launch_g(mmvq_s8_ic<4>{}); break;
                    case 5: launch_g(mmvq_s8_ic<5>{}); break;
                    case 6: launch_g(mmvq_s8_ic<6>{}); break;
                    case 7: launch_g(mmvq_s8_ic<7>{}); break;
                    case 8: launch_g(mmvq_s8_ic<8>{}); break;
                    default: GGML_ABORT("fatal error");
                }
            } else {
                // rows-outer kernel, 3 or 4 rows per warp (2..4 columns)
                const auto launch_ro = [&](auto ncols_tag, auto rows_tag) {
                    constexpr int c_ncols_dst    = decltype(ncols_tag)::value;
                    constexpr int rows_per_block = decltype(rows_tag)::value;
                    const dim3 block_nums((nrows_x + rows_per_block - 1)/rows_per_block, nchannels_dst, nsamples_dst);
                    const dim3 block_dims(warp_size, 1, 1);
                    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
                    ggml_cuda_kernel_launch(mul_mat_vec_q_s8<type, c_ncols_dst, false, rows_per_block>, launch_params,
                        vx, vy, fusion, dst, (uint32_t) ncols_x, (uint32_t) nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                        s8.stride_col_x_bias, s8.stride_col_gate_bias, channel_ratio_fd, stride_channel_x, stride_channel_y,
                        stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, s8.w_static);
                };
                const mmvq_s8_cfg cfg = mmvq_s8_select(type, ncols_dst, (int64_t) nrows_x*nchannels_dst*nsamples_dst);
                const auto launch_ng = [&](auto ncols_tag) {
                    constexpr int c_ncols = decltype(ncols_tag)::value;
                    if constexpr (c_ncols <= 4) {
                        switch (cfg) {
                            case MMVQ_S8_RO3: launch_ro(ncols_tag, mmvq_s8_ic<3>{}); break;
                            case MMVQ_S8_RO4: launch_ro(ncols_tag, mmvq_s8_ic<4>{}); break;
                            case MMVQ_S8_P4:  launch_w(ncols_tag, f_t{}, same_t{}, mmvq_s8_ic<4>{}, mmvq_s8_ic<1>{}, mmvq_s8_ic<12>{}, t_t{}); break;
                            case MMVQ_S8_K2:  launch_w(ncols_tag, f_t{}, same_t{}, mmvq_s8_ic<4>{}, mmvq_s8_ic<2>{}, mmvq_s8_ic<12>{}, f_t{}); break;
                            default:          launch_w(ncols_tag, f_t{}, same_t{}, mmvq_s8_ic<4>{}, mmvq_s8_ic<1>{}, mmvq_s8_ic<12>{}, f_t{}); break;
                        }
                    } else {
                        switch (cfg) {
                            case MMVQ_S8_K2:  launch_w(ncols_tag, f_t{}, same_t{}, mmvq_s8_ic<4>{}, mmvq_s8_ic<2>{}, mmvq_s8_ic<12>{}, f_t{}); break;
                            case MMVQ_S8_W14: launch_w(ncols_tag, f_t{}, same_t{}, mmvq_s8_ic<4>{}, mmvq_s8_ic<1>{}, mmvq_s8_ic<14>{}, f_t{}); break;
                            default:          launch_w(ncols_tag, f_t{}, same_t{}, mmvq_s8_ic<4>{}, mmvq_s8_ic<1>{}, mmvq_s8_ic<12>{}, f_t{}); break;
                        }
                    }
                };
                switch (ncols_dst) {
                    case 2: launch_ng(mmvq_s8_ic<2>{}); break;
                    case 3: launch_ng(mmvq_s8_ic<3>{}); break;
                    case 4: launch_ng(mmvq_s8_ic<4>{}); break;
                    case 5: launch_ng(mmvq_s8_ic<5>{}); break;
                    case 6: launch_ng(mmvq_s8_ic<6>{}); break;
                    case 7: launch_ng(mmvq_s8_ic<7>{}); break;
                    case 8: launch_ng(mmvq_s8_ic<8>{}); break;
                    default: GGML_ABORT("fatal error");
                }
            }
            return;
        }
    }
#else
    GGML_UNUSED(s8);
#endif // GGML_CUDA_MMVQ_S8_AVAILABLE

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2:
            launch_few_cols(std::integral_constant<int, 2>{});
            break;
        case 3:
            launch_few_cols(std::integral_constant<int, 3>{});
            break;
        case 4:
            launch_few_cols(std::integral_constant<int, 4>{});
            break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const mmvq_s8_args & s8 = {}) {
    // only the few-column kernel supports a gate matrix of another type, see ggml_cuda_mmvq_s8_can_fuse_mixed
    GGML_ASSERT(s8.type_gate == GGML_TYPE_COUNT || s8.type_gate == type_x || ggml_cuda_mmvq_s8_type(type_x));
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, s8);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const void * src1_q8_1_pre) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mmvq_s8_args s8_args;
    s8_args.w_static = ggml_cuda_mmvq_s8_weights_static(src0);

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        // several columns: 2D products in the few-column kernel only (fused biases are read with dst's channel strides)
        GGML_ASSERT(  ids || dst->ne[1] == 1 ||
                    (dst->ne[2] == 1 && dst->ne[3] == 1 && ggml_cuda_mmvq_s8_usable(src0->type, cc, dst->ne[1], false)));
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
            s8_args.stride_col_x_bias = fusion->x_bias->nb[1] / sizeof(float);
        }
        if (fusion->gate) {
            GGML_ASSERT((fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0)) ||
                        (!ids && !fusion->x_bias && !fusion->gate_bias && ggml_cuda_mmvq_s8_can_fuse_mixed(src0, fusion->gate, dst->ne[1])));
            fusion_local.gate = fusion->gate->data;
            s8_args.type_gate = fusion->gate->type;
            s8_args.w_static  = s8_args.w_static && ggml_cuda_mmvq_s8_weights_static(fusion->gate);
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
            s8_args.stride_col_gate_bias = fusion->gate_bias->nb[1] / sizeof(float);
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1_alloc(ctx.pool());
    const void * src1_q8_1 = src1_q8_1_pre;
    if (src1_q8_1 == nullptr) {
        src1_q8_1_alloc.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1_alloc.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        src1_q8_1 = src1_q8_1_alloc.get();
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream, s8_args);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mmvq_s8_args s8_args;
    s8_args.w_static = ggml_cuda_mmvq_s8_weights_static(src0);
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream, s8_args);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
