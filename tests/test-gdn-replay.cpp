// Rollback by replay (GGML_OP_GATED_DELTA_NET_REPLAY) against the snapshot scheme, bitwise, on one backend.
//
// A sequence of verify batches (n tokens, then a random number of them accepted) runs through
//   (A) the snapshot graph of the v2 runtime: l2norm(q/k) as rms_norm+scale, gather of the state row of the rollback
//       slot, gated_delta_net with K snapshots, snapshot cpy into the cache (the fused CUDA pattern), and
//   (B) the replay graph of llm_build_delta_net_base::build_recurrent_attn: the same norms, state and ring gathers,
//       gated_delta_net_replay, set_rows of the state and of the new ring rows, with the row indices and the fold
//       decisions of llama_memory_recurrent (one state + a circular buffer of Cap ring rows, default 2*W).
// The attention output of every batch and the final state must be identical bit for bit. Long batches (prompt
// chunks, the long kernel) are mixed in, also right after a rollback.
//
// usage: test-gdn-replay [backend name, default: first GPU, else CPU] [n_cycles] [W] [H_k] [v_rep] [seed] [n_layers] [Cap]
// With n_layers > 1 every graph holds that many independent layers (own caches, same inputs), so that the recurrent
// states do not stay in the L2 cache (48 layers of 3 MB for the production model): a kernel-time microbenchmark.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int64_t S = 128;

struct inputs {
    int64_t n;
    std::vector<float> qk, v, g, b;
};

struct runner {
    ggml_backend_t backend;
    int64_t H_k, H_v, W, D, F;
    int     L;

    ggml_context          * ctx_p = nullptr; // persistent caches
    ggml_backend_buffer_t   buf_p = nullptr;
    int64_t Cap;
    std::vector<ggml_tensor *> cache_a;  // [D, W]     snapshot slots (A), per layer
    std::vector<ggml_tensor *> cache_bs; // [D, 1]     state (B)
    std::vector<ggml_tensor *> cache_br; // [F, Cap]   ring buffer (B)

    ggml_gallocr_t galloc = nullptr;

    // bookkeeping
    int32_t a_row = 0;                    // A: rollback slot to read
    int32_t b_off = 0, b_cnt = 0, b_last = 0;
    int     n_fold = 0;
    int64_t n_replayed = 0; // ring rows replayed (all batches)

    runner(ggml_backend_t backend, int64_t H_k, int64_t v_rep, int64_t W, int L, int64_t cap) : backend(backend), H_k(H_k), H_v(H_k * v_rep), W(W), L(L) {
        D   = S * S * H_v;
        F   = S * H_k + S * H_v + 2 * H_v;
        Cap = std::max(cap, 2 * W);
        ggml_init_params ip = { 3 * (size_t) L * ggml_tensor_overhead(), nullptr, true };
        ctx_p = ggml_init(ip);
        for (int l = 0; l < L; ++l) {
            cache_a .push_back(ggml_new_tensor_2d(ctx_p, GGML_TYPE_F32, D, W));
            cache_bs.push_back(ggml_new_tensor_2d(ctx_p, GGML_TYPE_F32, D, 1));
            cache_br.push_back(ggml_new_tensor_2d(ctx_p, GGML_TYPE_F32, F, Cap));
        }
        buf_p    = ggml_backend_alloc_ctx_tensors(ctx_p, backend);
        ggml_backend_buffer_clear(buf_p, 0);
        galloc   = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }

    ~runner() {
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(buf_p);
        ggml_free(ctx_p);
    }

    void set_initial_state(const std::vector<float> & s0) {
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_set(cache_a[l],  s0.data(), 0, D * sizeof(float));
            ggml_backend_tensor_set(cache_bs[l], s0.data(), 0, D * sizeof(float));
        }
        a_row = 0;
        b_off = 0; b_cnt = 0; b_last = 0;
    }

    struct graph_in {
        ggml_tensor * qk, * v, * g, * b;
    };

    graph_in new_inputs(ggml_context * ctx, int64_t n) {
        graph_in gi;
        gi.qk = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S * 2 * H_k, n, 1);
        gi.v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, n, 1);
        gi.g  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, n, 1);
        gi.b  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, n, 1);
        ggml_set_input(gi.qk); ggml_set_input(gi.v); ggml_set_input(gi.g); ggml_set_input(gi.b);
        return gi;
    }

    void set_inputs(const graph_in & gi, const inputs & in) {
        ggml_backend_tensor_set(gi.qk, in.qk.data(), 0, ggml_nbytes(gi.qk));
        ggml_backend_tensor_set(gi.v,  in.v.data(),  0, ggml_nbytes(gi.v));
        ggml_backend_tensor_set(gi.g,  in.g.data(),  0, ggml_nbytes(gi.g));
        ggml_backend_tensor_set(gi.b,  in.b.data(),  0, ggml_nbytes(gi.b));
    }

    // l2 norms in the node order of the fused CUDA pattern
    void norms(ggml_context * ctx, ggml_cgraph * gf, const graph_in & gi, int64_t n, ggml_tensor ** q, ggml_tensor ** k) {
        const size_t nb1 = gi.qk->nb[1];
        ggml_tensor * rq = ggml_view_4d(ctx, gi.qk, S, H_k, n, 1, S * sizeof(float), nb1, nb1 * n, 0);
        ggml_tensor * rk = ggml_view_4d(ctx, gi.qk, S, H_k, n, 1, S * sizeof(float), nb1, nb1 * n, S * H_k * sizeof(float));
        ggml_build_forward_expand(gf, rq);
        ggml_build_forward_expand(gf, rk);
        const float eps = 1e-6f;
        *q = ggml_scale(ctx, ggml_rms_norm(ctx, rq, eps / S), 1.0f / sqrtf((float) S));
        ggml_build_forward_expand(gf, *q);
        *k = ggml_scale(ctx, ggml_rms_norm(ctx, rk, eps / S), 1.0f / sqrtf((float) S));
        ggml_build_forward_expand(gf, *k);
    }

    // (A) snapshot scheme; returns the attention output
    std::vector<float> run_a(const inputs & in, int rollback_after) {
        const int64_t n = in.n;
        ggml_init_params ip = { (64 + 48 * (size_t) L) * ggml_tensor_overhead() + ggml_graph_overhead_custom(64 * (size_t) L + 64, false), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph  * gf  = ggml_new_graph_custom(ctx, 64 * (size_t) L + 64, false);

        graph_in gi = new_inputs(ctx, n);
        ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(rows);

        std::vector<ggml_tensor *> outs;
        for (int l = 0; l < L; ++l) {
            ggml_tensor * q, * k;
            norms(ctx, gf, gi, n, &q, &k);
            ggml_tensor * st = ggml_reshape_4d(ctx, ggml_get_rows(ctx, cache_a[l], rows), S, S, H_v, 1);
            ggml_build_forward_expand(gf, st);
            ggml_tensor * gdn = ggml_gated_delta_net(ctx, q, k, gi.v, gi.g, gi.b, st, W);
            const int64_t n_written = std::min<int64_t>(n, W);
            ggml_tensor * tail = ggml_view_3d(ctx, gdn, D, 1, n_written, D * sizeof(float), D * sizeof(float), S * H_v * n * sizeof(float));
            ggml_tensor * dst  = ggml_view_3d(ctx, cache_a[l], D, 1, n_written, cache_a[l]->nb[1], cache_a[l]->nb[1], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, dst));
            ggml_tensor * attn = ggml_view_4d(ctx, gdn, S, H_v, n, 1, S * sizeof(float), S * H_v * sizeof(float), S * H_v * n * sizeof(float), 0);
            ggml_tensor * out  = ggml_scale(ctx, attn, 1.0f);
            ggml_set_output(out);
            ggml_build_forward_expand(gf, out);
            outs.push_back(out);
        }

        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
        set_inputs(gi, in);
        ggml_backend_tensor_set(rows, &a_row, 0, sizeof(int32_t));
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        std::vector<float> res;
        for (ggml_tensor * out : outs) {
            std::vector<float> o(ggml_nelements(out));
            ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));
            res.insert(res.end(), o.begin(), o.end());
        }
        ggml_free(ctx);

        // slot r = state r tokens before the end of the batch
        a_row = rollback_after;
        return res;
    }

    // (B) rollback by replay; returns the attention output
    std::vector<float> run_b(const inputs & in, int rollback_after) {
        const int64_t n     = in.n;
        const bool    chain = n <= W;
        const int64_t n_rn  = chain ? n : 0;
        ggml_init_params ip = { (64 + 48 * (size_t) L) * ggml_tensor_overhead() + ggml_graph_overhead_custom(64 * (size_t) L + 64, false), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph  * gf  = ggml_new_graph_custom(ctx, 64 * (size_t) L + 64, false);

        graph_in gi = new_inputs(ctx, n);
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 + Cap + n_rn);
        ggml_set_input(idx);
        const size_t es = sizeof(int32_t);
        ggml_tensor * i_s_src    = ggml_view_1d(ctx, idx, 1,    0);
        ggml_tensor * i_rep      = ggml_view_1d(ctx, idx, 2,    1 * es);
        ggml_tensor * i_ring_src = ggml_view_1d(ctx, idx, Cap,  3 * es);
        ggml_tensor * i_s_dst    = ggml_view_1d(ctx, idx, 1,    (3 + Cap) * es);
        ggml_tensor * i_ring_dst = ggml_view_1d(ctx, idx, n_rn, (4 + Cap) * es);

        std::vector<ggml_tensor *> outs;
        for (int l = 0; l < L; ++l) {
            ggml_tensor * q, * k;
            norms(ctx, gf, gi, n, &q, &k);
            ggml_tensor * st = ggml_reshape_4d(ctx, ggml_get_rows(ctx, cache_bs[l], i_s_src), S, S, H_v, 1);
            ggml_build_forward_expand(gf, st);
            ggml_tensor * rg = ggml_reshape_3d(ctx, ggml_get_rows(ctx, cache_br[l], i_ring_src), F, Cap, 1);
            ggml_build_forward_expand(gf, rg);
            ggml_tensor * op = ggml_gated_delta_net_replay(ctx, q, k, gi.v, gi.g, gi.b, st, rg, i_rep, (int32_t) W);
            const size_t eo  = sizeof(float);
            const size_t off = (size_t) (S * H_v * n) * eo;
            ggml_tensor * attn = ggml_view_4d(ctx, op, S, H_v, n, 1, S * eo, S * H_v * eo, S * H_v * n * eo, 0);
            ggml_tensor * sout = ggml_view_2d(ctx, op, D, 1, D * eo, off);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_bs[l], sout, i_s_dst));
            if (chain) {
                ggml_tensor * rnew = ggml_view_2d(ctx, op, F, n, F * eo, off + D * eo);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_br[l], rnew, i_ring_dst));
            }
            ggml_tensor * out = ggml_scale(ctx, attn, 1.0f);
            ggml_set_output(out);
            ggml_build_forward_expand(gf, out);
            outs.push_back(out);
        }

        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
        set_inputs(gi, in);

        // llama_memory_recurrent::find_slot + set_input_rs_replay for one cell that does not move
        const int32_t off0 = b_off, cnt0 = b_cnt;
        const int32_t fold = !chain || cnt0 + n + W > Cap || n > 8 ? 1 : 0;
        n_fold += fold;
        n_replayed += cnt0;
        std::vector<int32_t> d;
        d.push_back(0);
        d.push_back(cnt0);
        d.push_back(fold);
        for (int32_t t = 0; t < Cap; ++t) {
            d.push_back((off0 + t) % Cap);
        }
        d.push_back(0);
        for (int32_t t = 0; t < n_rn; ++t) {
            d.push_back((off0 + cnt0 + t) % Cap);
        }
        ggml_backend_tensor_set(idx, d.data(), 0, ggml_nbytes(idx));
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        std::vector<float> res;
        for (ggml_tensor * out : outs) {
            std::vector<float> o(ggml_nelements(out));
            ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));
            res.insert(res.end(), o.begin(), o.end());
        }
        ggml_free(ctx);

        if (!chain) {
            b_off = 0; b_cnt = 0; b_last = 0;
        } else {
            b_off  = fold ? (off0 + cnt0) % Cap : off0;
            b_cnt  = fold ? (int32_t) n : cnt0 + (int32_t) n;
            b_last = (int32_t) n;
        }
        if (rollback_after > 0) {
            GGML_ASSERT(rollback_after <= b_last);
            b_cnt -= rollback_after;
            b_last = 0;
        }
        return res;
    }

    std::vector<float> state_a() {
        std::vector<float> s(D * L);
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_get(cache_a[l], s.data() + l * D, (size_t) a_row * D * sizeof(float), D * sizeof(float));
        }
        return s;
    }

    // only valid without pending rows
    std::vector<float> state_b() {
        GGML_ASSERT(b_cnt == 0);
        std::vector<float> s(D * L);
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_get(cache_bs[l], s.data() + l * D, 0, D * sizeof(float));
        }
        return s;
    }
};

static inputs make_inputs(std::mt19937 & rng, int64_t n, int64_t H_k, int64_t H_v) {
    std::uniform_real_distribution<float> du(-1.0f, 1.0f);
    inputs in;
    in.n = n;
    in.qk.resize(S * 2 * H_k * n);
    in.v.resize(S * H_v * n);
    in.g.resize(H_v * n);
    in.b.resize(H_v * n);
    for (auto & x : in.qk) { x = 2.0f * du(rng); }
    for (auto & x : in.v)  { x = -0.3f + 2.65f * (du(rng) + 1.0f); }
    for (auto & x : in.g)  { x = -10.0f + 9.99f * du(rng); }
    for (auto & x : in.b)  { x = 0.5f + 0.5f * du(rng); }
    return in;
}

static size_t n_diff(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    return n;
}

int main(int argc, char ** argv) {
    const char * dev_name = argc > 1 ? argv[1] : nullptr;
    const int    n_cycles = argc > 2 ? atoi(argv[2]) : 400;
    const int    W        = argc > 3 ? atoi(argv[3]) : 4;
    const int    H_k      = argc > 4 ? atoi(argv[4]) : 16;
    const int    v_rep    = argc > 5 ? atoi(argv[5]) : 3;
    const int    seed     = argc > 6 ? atoi(argv[6]) : 1234;
    const int    n_layers = argc > 7 ? atoi(argv[7]) : 1;
    const int    cap      = argc > 8 ? atoi(argv[8]) : 2 * W;

    ggml_backend_load_all();

    ggml_backend_t backend = nullptr;
    if (dev_name) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
        if (!dev) {
            fprintf(stderr, "unknown device %s\n", dev_name);
            return 1;
        }
        backend = ggml_backend_dev_init(dev, nullptr);
    } else {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
    }
    GGML_ASSERT(backend);
    printf("backend %s, %d cycles, W = %d, H_k = %d, H_v = %d, seed %d, %d layers, Cap = %d\n", ggml_backend_name(backend), n_cycles, W, H_k, H_k * v_rep, seed, n_layers, std::max(cap, 2 * W));

    runner r(backend, H_k, v_rep, W, n_layers, cap);
    std::mt19937 rng(seed);

    std::vector<float> s0(r.D);
    {
        std::uniform_real_distribution<float> du(-0.3f, 0.3f);
        for (auto & x : s0) { x = du(rng); }
    }
    r.set_initial_state(s0);

    size_t n_bad = 0, n_rollbacks = 0, n_long = 0, n_tokens = 0;
    int prev_rollback = 0;
    for (int c = 0; c < n_cycles; ++c) {
        // mostly verify batches of 1..W tokens; every 16th batch a long one (a prompt chunk), which is never rolled
        // back by the server but may follow a rollback
        // verify batches have at most 8 tokens: the snapshot scheme switches to the long kernel (other arithmetic)
        // above that, so replaying such a batch is exact only up to rounding
        const bool    is_long = c % 16 == 15;
        const int64_t n       = is_long ? std::max(W, 8) + 1 + (int64_t) (rng() % 40) : 1 + (int64_t) (rng() % std::min(W, 8));
        int rollback = 0;
        if (!is_long && n > 1) {
            const int accepted = 1 + (int) (rng() % n); // tokens kept of this batch (>= 1)
            rollback = std::min((int) n - accepted, W - 1);
        }
        const inputs in = make_inputs(rng, n, H_k, (int64_t) H_k * v_rep);
        const std::vector<float> oa = r.run_a(in, rollback);
        const std::vector<float> ob = r.run_b(in, rollback);
        const size_t nd = n_diff(oa, ob);
        if (nd != 0) {
            if (n_bad < 10) {
                fprintf(stderr, "cycle %d: n = %lld, previous rollback %d: %zu of %zu output values differ\n",
                        c, (long long) n, prev_rollback, nd, oa.size());
            }
            n_bad++;
        }
        n_rollbacks += rollback > 0;
        n_long      += is_long;
        n_tokens    += n;
        prev_rollback = rollback;
    }

    // a long batch so that B stores its final state, then compare the states
    {
        const inputs in = make_inputs(rng, W + 9, H_k, (int64_t) H_k * v_rep);
        const size_t nd = n_diff(r.run_a(in, 0), r.run_b(in, 0));
        const size_t ns = n_diff(r.state_a(), r.state_b());
        if (nd != 0 || ns != 0) {
            fprintf(stderr, "final: %zu output values and %zu state values differ\n", nd, ns);
            n_bad++;
        }
    }

    printf("%zu tokens in %d batches (%zu rolled back, %zu long, %d state writes by folds or long batches, %.2f rows replayed per batch): %s\n",
           n_tokens, n_cycles, n_rollbacks, n_long, r.n_fold, (double) r.n_replayed / (n_cycles + 1), n_bad == 0 ? "bitwise identical" : "MISMATCH");

    ggml_backend_free(backend);
    return n_bad == 0 ? 0 : 1;
}
