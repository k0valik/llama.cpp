#pragma once

#include "llama.h"
#include "common.h"
#include "sampling.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // candidate distribution per drafted token; set it to make draft-simple, draft-mtp and draft-dflash (DFlash2 only) sample
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    // coupled sampling (common_sampler_get_coupled): the target draws the j-th draft position with k = coupled.k + j;
    // drafters that support it propose with the same noise (coupled.temp = 0: disabled)
    common_sampler_coupled_params coupled;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// Acceptance-adaptive draft length (--spec-draft-adaptive, fused MTP drafting), one per sequence. Each cycle drafts
// n_draft() tokens; n_verify() then keeps the prefix whose expected number of emitted tokens per unit of time is highest,
// using the drafter's per-token confidences, and the next cycle drafts one token more than that prefix unless even a
// most confident extra token would not pay (so the length grows while the whole draft keeps paying for itself, and
// shrinks to what paid otherwise; the first cycle drafts min(3, n_max)). Online estimates:
//   - P(accept | position, confidence): decayed counts per (position 1 / 2 / 3+, confidence bin), from the verified
//     tokens up to the first rejection;
//   - cycle cost = draft time (measured each cycle) + rest(n_v): the rest of the cycle (verify of n_v + 1 tokens,
//     sampling, server) = cycle time - draft time, modelled as a line in n_v (increasing: a longer verify never looks
//     cheaper) whose level and slope are learned from every cycle with robust sign updates (see observe_time), plus
//     the learned cost of verifying another number of tokens than the previous cycle (graph switch), which both
//     decisions pay when they change the length.
struct common_spec_adaptive {
    static constexpr int N_POS = 3;
    static constexpr int N_BIN = 11;
    static constexpr int N_MAX = 16;

    explicit common_spec_adaptive(int32_t n_max = 7);

    int32_t n_draft() const { return n_next; }

    // how many of the n drafted tokens (confidences conf[0..n)) to verify (1..n); t_draft_ms = time of this draft
    int32_t n_verify(const float * conf, int32_t n, double t_draft_ms);

    // the target accepted n_acc of the n_v verified tokens with confidences conf[0..n_v)
    void observe_accept(const float * conf, int32_t n_v, int32_t n_acc);

    // a complete cycle that drafted n_d and verified n_v tokens took t_cycle_ms, t_draft_ms of it drafting
    void observe_time(int32_t n_d, int32_t n_v, double t_draft_ms, double t_cycle_ms);

    float  p_accept(int32_t j, float conf) const; // j = 0-based draft position
    double t_rest(int32_t n_v) const;
    double t_draft_model(int32_t n) const;

    int32_t n_max;
    int32_t n_next;

    float acc_s[N_POS][N_BIN]; // decayed accepted counts
    float acc_n[N_POS][N_BIN]; // decayed verified counts

    // rest(n_v) = tr_c + tr_b*(n_v - tr_x); priors measured on RTX 5070 Ti / Qwen3.8-27B IQ3_S: draft 0.17 + 0.61 ms per
    // step, rest 17.2 + 1.08 ms per verified draft token (short context)
    double  td_step = 0.62;         // ms per draft step
    double  tr_c    = 20.0;         // ms, rest at n_v = tr_x
    double  tr_b    = 1.1;          // ms per verified draft token
    double  tr_x    = 3.0;          // reference point: running mean of n_v
    double  tr_sw   = 1.0;          // ms, extra time of a cycle whose n_v differs from the previous one (graph switch)
    int32_t n_time  = 0;            // cycle times observed
    int32_t nv_last = -1;           // last n_verify() result
    int32_t nv_obs  = -1;           // n_v of the last observed cycle
};

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// A failed decode can leave shared draft state partially modified. Once false, the
// speculative context must be destroyed and reinitialized before it can be used again.
bool common_speculative_is_healthy(const common_speculative * spec);

// Generate drafts. The checked form reports failure; the compatibility wrapper throws
// on failure rather than silently allowing callers to continue with inconsistent state.
bool common_speculative_draft_checked(common_speculative * spec);
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state. Only one stateful implementation can be
// checkpointed; invalid MTP state throws and invalidates the speculator.
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
bool common_speculative_mtp_state_valid(const common_speculative * spec, const std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
