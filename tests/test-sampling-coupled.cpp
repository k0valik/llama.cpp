// Statistical check of the coupled (Gumbel-max) final sampler of common/sampling.cpp:
// a fixed logit vector goes through top-k 20 / top-p 0.95 / temp, then the final sampler. The empirical distribution
// over many (seed, k) must match the softmax over the surviving candidates, like the stock dist sampler.
// spec_sim: speculative decoding with drafters that use the same noise must emit the same, exactly distributed tokens.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "../src/llama-graph.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

struct stats {
    double chi2   = 0.0;
    int    dof    = 0;
    double max_ae = 0.0; // max |freq - p|
    double max_z  = 0.0; // max |freq - p| / sigma
};

// upper tail of the chi-square distribution (Wilson-Hilferty)
static double chi2_pvalue(double x, int k) {
    const double z = (std::cbrt(x/k) - (1.0 - 2.0/(9.0*k)))/std::sqrt(2.0/(9.0*k));
    return 0.5*std::erfc(z/std::sqrt(2.0));
}

static stats compare(const std::map<llama_token, double> & p, const std::map<llama_token, int64_t> & cnt, int64_t n) {
    stats s;
    for (const auto & [id, pi] : p) {
        const auto it = cnt.find(id);
        const double o = it == cnt.end() ? 0.0 : (double) it->second;
        const double e = pi*n;
        s.chi2 += (o - e)*(o - e)/e;
        const double ae = std::fabs(o/n - pi);
        s.max_ae = std::max(s.max_ae, ae);
        s.max_z  = std::max(s.max_z, ae/std::sqrt(pi*(1.0 - pi)/n));
    }
    s.dof = (int) p.size() - 1;
    for (const auto & [id, c] : cnt) {
        GGML_ASSERT(p.count(id) && "sampled a token outside of the candidates");
    }
    return s;
}

static llama_sampler * make_chain(float temp, llama_sampler * last) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    if (last) {
        llama_sampler_chain_add(chain, last);
    }
    return chain;
}

// fixed logits over 2000 tokens with ids spread over the vocab
static std::vector<llama_token_data> make_logits(float spread) {
    std::vector<llama_token_data> base(2000);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, spread);
    for (size_t i = 0; i < base.size(); ++i) {
        base[i] = { (llama_token) ((int64_t) i*7919 % 248320), nd(rng), 0.0f };
    }
    return base;
}

static llama_token_data_array apply(llama_sampler * chain, const std::vector<llama_token_data> & base, std::vector<llama_token_data> & cur) {
    cur = base;
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(chain, &cur_p);
    return cur_p;
}

static bool test_distribution(const std::vector<llama_token_data> & base, float spread, float temp, int64_t n_samples) {
    std::vector<llama_token_data> cur;

    // expected distribution: softmax over what the truncation keeps
    std::map<llama_token, double> p;
    {
        llama_sampler * chain = make_chain(temp, nullptr);
        const auto cur_p = apply(chain, base, cur);
        double mx = -INFINITY, sum = 0.0;
        for (size_t i = 0; i < cur_p.size; ++i) {
            mx = std::max(mx, (double) cur_p.data[i].logit);
        }
        for (size_t i = 0; i < cur_p.size; ++i) {
            sum += std::exp((double) cur_p.data[i].logit - mx);
        }
        for (size_t i = 0; i < cur_p.size; ++i) {
            p[cur_p.data[i].id] = std::exp((double) cur_p.data[i].logit - mx)/sum;
        }
        llama_sampler_free(chain);
    }

    // stock dist sampler
    std::map<llama_token, int64_t> cnt_dist;
    {
        llama_sampler * chain = make_chain(temp, llama_sampler_init_dist(1234));
        for (int64_t s = 0; s < n_samples; ++s) {
            const auto cur_p = apply(chain, base, cur);
            cnt_dist[cur_p.data[cur_p.selected].id]++;
        }
        llama_sampler_free(chain);
    }

    // coupled sampler: 1000 draws (k = 0..999) per seed
    std::map<llama_token, int64_t> cnt_cpl;
    std::map<llama_token, int64_t> cnt_sub; // second draw at the same k (grammar resampling)
    int64_t n_same_sub  = 0; // first draw == second draw at the same k
    int64_t n_same_next = 0; // draw at k == draw at k + 1
    {
        const int64_t n_per_seed = 1000;
        llama_token prev = LLAMA_TOKEN_NULL;
        llama_sampler * chain = nullptr;
        for (int64_t s = 0; s < n_samples; ++s) {
            if (s % n_per_seed == 0) {
                llama_sampler_free(chain);
                chain = make_chain(temp, common_sampler_init_coupled((uint32_t) (s/n_per_seed)*2654435761u + 17));
                prev = LLAMA_TOKEN_NULL;
            }
            const auto cur_p = apply(chain, base, cur);
            const llama_token id = cur_p.data[cur_p.selected].id;
            cnt_cpl[id]++;

            n_same_next += id == prev;
            prev = id;

            const auto cur_p2 = apply(chain, base, cur);
            const llama_token id2 = cur_p2.data[cur_p2.selected].id;
            cnt_sub[id2]++;
            n_same_sub += id == id2;

            llama_sampler_accept(chain, id);
        }
        llama_sampler_free(chain);
    }

    double sum_p2 = 0.0;
    for (const auto & [id, pi] : p) {
        sum_p2 += pi*pi;
    }

    const stats sd = compare(p, cnt_dist, n_samples);
    const stats sc = compare(p, cnt_cpl,  n_samples);
    const stats ss = compare(p, cnt_sub,  n_samples);

    printf("logit spread %.1f, temp %.2f: %zu candidates, %lld samples each\n", spread, temp, p.size(), (long long) n_samples);
    printf("  dist    : chi2 %8.2f (dof %d, p-value %.3f)  max|f-p| %.2e  max z %.2f\n", sd.chi2, sd.dof, chi2_pvalue(sd.chi2, sd.dof), sd.max_ae, sd.max_z);
    printf("  coupled : chi2 %8.2f (dof %d, p-value %.3f)  max|f-p| %.2e  max z %.2f\n", sc.chi2, sc.dof, chi2_pvalue(sc.chi2, sc.dof), sc.max_ae, sc.max_z);
    printf("  resample: chi2 %8.2f (dof %d, p-value %.3f)  max|f-p| %.2e  max z %.2f\n", ss.chi2, ss.dof, chi2_pvalue(ss.chi2, ss.dof), ss.max_ae, ss.max_z);
    printf("  P(draw == resample at same k) %.5f, P(draw k == draw k+1) %.5f, independent: %.5f\n",
            (double) n_same_sub/n_samples, (double) n_same_next/(n_samples - n_samples/1000), sum_p2);

    bool ok = true;
    for (const stats * st : { &sd, &sc, &ss }) {
        ok &= chi2_pvalue(st->chi2, st->dof) > 1e-4;
    }
    const double sigma = std::sqrt(sum_p2*(1.0 - sum_p2)/n_samples);
    ok &= std::fabs((double) n_same_sub/n_samples - sum_p2) < 5*sigma;

    return ok;
}

// greedy stays greedy
static bool test_greedy(const std::vector<llama_token_data> & base) {
    llama_sampler * chain = make_chain(0.0f, common_sampler_init_coupled(7));
    const auto best = std::max_element(base.begin(), base.end(), [](const auto & a, const auto & b) { return a.logit < b.logit; })->id;
    std::vector<llama_token_data> cur;
    bool ok = true;
    for (int s = 0; s < 1000; ++s) {
        const auto cur_p = apply(chain, base, cur);
        ok &= cur_p.data[cur_p.selected].id == best;
        llama_sampler_accept(chain, best);
    }
    llama_sampler_free(chain);
    printf("greedy (temp 0): %s\n", ok ? "ok" : "FAILED");
    return ok;
}

// Speculative decoding simulation. The "target" is a Markov chain over 64 tokens (logits depend on the last token) sampled
// through penalties / top-k / top-p / min-p / temp / coupled, with a grammar-like constraint that forces the resampling
// path (second draw at the same k) at some positions. Every cycle a drafter proposes up to 3 tokens (the random and the
// adaptive drafter up to 7; the adaptive one picks its length with common_spec_adaptive), the target verifies
// them like common_sampler_sample_and_accept_n (optionally followed by the server's checkpoint replay: restore the sampler
// from a clone and verify the accepted tokens again). Checks:
// - the emitted sequence is the same for every drafter and without speculation, including drafters that read the noise
//   of future positions (the draft only decides how many tokens a cycle emits);
// - the randomized PIT values of the emitted tokens under their exact conditional distributions are i.i.d. U(0, 1)
//   (single values and consecutive pairs, over all positions and cycle boundaries).
namespace spec_sim {

constexpr int   n_tok  = 64;
constexpr int   n_gen  = 48;
constexpr int   n_draft_max = 3;
constexpr float temp   = 0.8f;
constexpr float top_p  = 0.9f;
constexpr int   top_k  = 20;

struct model {
    std::vector<llama_token> id_of;
    std::vector<float> tgt; // [n_tok][n_tok] logits of the next token given the last one
    std::vector<float> dft; // a perturbed copy for the drafter

    model() : id_of(n_tok), tgt(n_tok*n_tok), dft(n_tok*n_tok) {
        std::mt19937 rng(4242);
        std::normal_distribution<float> nd(0.0f, 1.5f), pert(0.0f, 0.7f);
        for (int v = 0; v < n_tok; ++v) {
            id_of[v] = (llama_token) ((int64_t) v*3877 + 11) % 248320;
        }
        for (auto & x : tgt) {
            x = nd(rng);
        }
        for (size_t i = 0; i < dft.size(); ++i) {
            dft[i] = tgt[i] + pert(rng);
        }
    }

    int index(llama_token id) const {
        return (int) (std::find(id_of.begin(), id_of.end(), id) - id_of.begin());
    }

    // the grammar-like constraint: after some tokens, ids divisible by 5 are forbidden
    bool forbidden(llama_token last, llama_token id) const {
        return index(last) % 3 == 0 && id % 5 == 0;
    }

    std::vector<llama_token_data> logits(llama_token last, bool constrained) const {
        std::vector<llama_token_data> cur(n_tok);
        const int r = index(last);
        for (int v = 0; v < n_tok; ++v) {
            const bool masked = constrained && forbidden(last, id_of[v]);
            cur[v] = { id_of[v], masked ? -INFINITY : tgt[r*n_tok + v], 0.0f };
        }
        return cur;
    }
};

static llama_sampler * make_chain(llama_sampler * last) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(248320, 16, 1.2f, 0.05f, 0.1f));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    if (last) {
        llama_sampler_chain_add(chain, last);
    }
    return chain;
}

// like common_sampler_sample with grammar_first = false: draw, and if the grammar rejects the token, draw again from the
// constrained logits (the coupled sampler then uses the second stream of the same k)
static llama_token sample(const model & m, llama_sampler * chain, llama_token last) {
    auto cur = m.logits(last, false);
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(chain, &cur_p);
    const llama_token id = cur_p.data[cur_p.selected].id;
    if (!m.forbidden(last, id)) {
        return id;
    }
    cur = m.logits(last, true);
    cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(chain, &cur_p);
    return cur_p.data[cur_p.selected].id;
}

// exact probabilities of the tokens (by index) that sample() returns, given the chain state (ref has no final sampler)
static std::vector<double> exact_probs(const model & m, llama_sampler * ref, llama_token last) {
    auto softmax = [&](bool constrained) {
        auto cur = m.logits(last, constrained);
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(ref, &cur_p);
        std::vector<double> p(n_tok, 0.0);
        double mx = -INFINITY, sum = 0.0;
        for (size_t i = 0; i < cur_p.size; ++i) {
            mx = std::max(mx, (double) cur_p.data[i].logit);
        }
        for (size_t i = 0; i < cur_p.size; ++i) {
            if (cur_p.data[i].logit != -INFINITY) {
                const double e = std::exp((double) cur_p.data[i].logit - mx);
                p[m.index(cur_p.data[i].id)] = e;
                sum += e;
            }
        }
        for (auto & x : p) {
            x /= sum;
        }
        return p;
    };
    auto p = softmax(false);
    double p_bad = 0.0;
    for (int v = 0; v < n_tok; ++v) {
        if (m.forbidden(last, m.id_of[v])) {
            p_bad += p[v];
            p[v] = 0.0;
        }
    }
    if (p_bad > 0.0) {
        const auto q = softmax(true);
        for (int v = 0; v < n_tok; ++v) {
            p[v] += p_bad*q[v];
        }
    }
    return p;
}

enum drafter_type { DRAFT_NONE, DRAFT_COUPLED, DRAFT_NOISE, DRAFT_ORACLE, DRAFT_RANDOM, DRAFT_ADAPTIVE };

constexpr int n_draft_long = 7; // draft length limit of the random and adaptive drafters

// the MTP drafter: its own logits, truncated like the target, argmax with the noise of k + j; returns the pick and its
// confidence (its softmax share of the perturbed logits, like the fused draft graph)
static std::pair<llama_token, float> draft_coupled(const model & m, uint32_t seed, uint32_t k, int j, llama_token prev) {
    const int r = m.index(prev);
    std::vector<std::pair<float, int>> c(n_tok);
    for (int v = 0; v < n_tok; ++v) {
        c[v] = { m.dft[r*n_tok + v], v };
    }
    std::sort(c.begin(), c.end(), std::greater<>());
    c.resize(top_k);
    double sum = 0.0, cum = 0.0;
    for (const auto & x : c) {
        sum += std::exp((double) x.first - c[0].first);
    }
    size_t n = c.size();
    for (size_t i = 0; i < c.size(); ++i) {
        cum += std::exp((double) c[i].first - c[0].first);
        if (cum >= top_p*sum) {
            n = i + 1;
            break;
        }
    }
    int best = c[0].second;
    float best_v = -INFINITY;
    std::vector<float> vals(n);
    for (size_t i = 0; i < n; ++i) {
        vals[i] = c[i].first/temp + ggml_gumbel_noise(seed, k + j, m.id_of[c[i].second]);
        if (vals[i] > best_v) {
            best_v = vals[i];
            best   = c[i].second;
        }
    }
    double z = 0.0;
    for (const float v : vals) {
        z += std::exp((double) v - best_v);
    }
    return { m.id_of[best], (float) (1.0/z) };
}

struct generation {
    std::vector<llama_token> out;
    int n_cycles = 0;
    int n_accepted = 0;
    int n_len[n_draft_long + 1] = {}; // cycles by draft length
};

static generation generate(const model & m, uint32_t seed, drafter_type dt, bool replay) {
    llama_sampler * chain = make_chain(common_sampler_init_coupled(seed));
    std::mt19937 rng(seed);

    // the adaptive draft length policy (kept across generations like a server slot's), fed with synthetic cycle times
    static common_spec_adaptive adapt_by_parity[2] = { common_spec_adaptive(n_draft_long), common_spec_adaptive(n_draft_long) };
    common_spec_adaptive & adapt = adapt_by_parity[seed % 2];

    // prompt: accepted like server_slot::init_sampler (k advances)
    uint32_t k = 0;
    llama_token last = 0;
    for (int i = 0; i < 5; ++i) {
        last = m.id_of[rng() % n_tok];
        llama_sampler_accept(chain, last);
        k++;
    }

    generation g;
    while ((int) g.out.size() < n_gen) {
        std::vector<llama_token> draft;
        std::vector<float> conf;
        switch (dt) {
            case DRAFT_NONE:
                break;
            case DRAFT_COUPLED: {
                llama_token prev = last;
                for (int j = 0; j < n_draft_max; ++j) {
                    prev = draft_coupled(m, seed, k, j, prev).first;
                    draft.push_back(prev);
                }
            } break;
            case DRAFT_ADAPTIVE: {
                // like the fused MTP drafter with --spec-draft-adaptive: draft n_draft() tokens, verify a prefix
                const int n_d = adapt.n_draft();
                llama_token prev = last;
                llama_sampler * sim = seed % 2 == 0 ? llama_sampler_clone(chain) : nullptr;
                for (int j = 0; j < n_d; ++j) {
                    // even seeds: a near-perfect drafter (simulates the target, wrong every 16th token), so that long
                    // drafts pay off; odd seeds: the MTP-like drafter
                    auto [t, c] = sim ? std::make_pair(sample(m, sim, prev), 0.999f) : draft_coupled(m, seed, k, j, prev);
                    if (sim) {
                        llama_sampler_accept(sim, t);
                        if (rng() % 16 == 0) {
                            t = m.id_of[(m.index(t) + 1) % n_tok];
                        }
                    }
                    draft.push_back(t);
                    conf.push_back(c);
                    prev = t;
                }
                if (sim) {
                    llama_sampler_free(sim);
                }
                draft.resize(adapt.n_verify(conf.data(), n_d, (seed % 2 ? 0.9 : 0.1)*n_d));
            } break;
            case DRAFT_NOISE: {
                // adversarial: pure noise of the future positions, no logits at all
                for (int j = 0; j < n_draft_max; ++j) {
                    int best = 0;
                    float best_v = -INFINITY;
                    for (int v = 0; v < n_tok; ++v) {
                        const float gv = ggml_gumbel_noise(seed, k + j, m.id_of[v]);
                        if (gv > best_v) {
                            best_v = gv;
                            best   = v;
                        }
                    }
                    draft.push_back(m.id_of[best]);
                }
            } break;
            case DRAFT_ORACLE: {
                // adversarial: simulates the target exactly (same chain state and noise) and uses that knowledge to
                // accept only the low-probability picks, i.e. it chooses the cycle lengths from the future noise
                llama_sampler * sim = llama_sampler_clone(chain);
                llama_token prev = last;
                for (int j = 0; j < n_draft_max; ++j) {
                    const llama_token t = sample(m, sim, prev);
                    llama_sampler_accept(sim, t);
                    const bool rare = (ggml_gumbel_noise(seed, k + j, t) > 1.0f);
                    const llama_token d = rare ? t : m.id_of[(m.index(t) + 1) % n_tok];
                    draft.push_back(d);
                    if (d != t) {
                        break;
                    }
                    prev = t;
                }
                llama_sampler_free(sim);
            } break;
            case DRAFT_RANDOM: {
                const int n = 1 + rng() % n_draft_long;
                for (int j = 0; j < n; ++j) {
                    draft.push_back(m.id_of[rng() % n_tok]);
                }
            } break;
        }

        // verify (common_sampler_sample_and_accept_n)
        llama_sampler * save = llama_sampler_clone(chain);
        auto verify = [&](const std::vector<llama_token> & d) {
            std::vector<llama_token> acc;
            llama_token prev = last;
            size_t i = 0;
            for (; i < d.size(); ++i) {
                const llama_token t = sample(m, chain, prev);
                llama_sampler_accept(chain, t);
                acc.push_back(t);
                if (d[i] != t) {
                    break;
                }
                prev = t;
            }
            if (i == d.size()) {
                const llama_token t = sample(m, chain, prev);
                llama_sampler_accept(chain, t);
                acc.push_back(t);
            }
            return acc;
        };
        auto acc = verify(draft);
        if (replay && acc.size() < draft.size() + 1) {
            // server: partial acceptance with a checkpoint -> restore the sampler, verify the accepted tokens again
            llama_sampler_copy(save, chain);
            acc = verify(acc);
        }
        llama_sampler_free(save);

        if (dt == DRAFT_ADAPTIVE) {
            const int n_v = (int) draft.size();
            adapt.observe_accept(conf.data(), n_v, (int) acc.size() - 1);
            // even seeds: cheap extra tokens (the lengths grow to the limit), odd seeds: ~1 ms per token
            const double c = seed % 2 ? 0.9 : 0.1;
            adapt.observe_time((int) conf.size(), n_v, c*conf.size(), c*conf.size() + 19.0 + c*n_v + 0.3*(rng() % 3));
        }

        g.n_cycles++;
        g.n_len[std::min<size_t>(draft.size(), n_draft_long)]++;
        g.n_accepted += (int) acc.size() - 1;
        k += (uint32_t) acc.size();
        for (const llama_token t : acc) {
            g.out.push_back(t);
        }
        last = acc.back();
    }
    g.out.resize(n_gen);

    llama_sampler_free(chain);
    return g;
}

static bool test(int n_seeds) {
    const model m;

    bool ok = true;

    int64_t n_bins[20] = {};
    int64_t n_pair[5][5] = {};
    int64_t n_u = 0, n_p = 0;
    int64_t n_cycles[6] = {}, n_acc[6] = {};
    int64_t n_len_adaptive[n_draft_long + 1] = {};
    int n_diff = 0;

    for (int s = 0; s < n_seeds; ++s) {
        const uint32_t seed = (uint32_t) s*2654435761u + 99;

        const generation ref = generate(m, seed, DRAFT_NONE, false);
        for (int dt = DRAFT_COUPLED; dt <= DRAFT_ADAPTIVE; ++dt) {
            for (bool replay : { false, true }) {
                const generation g = generate(m, seed, (drafter_type) dt, replay);
                if (g.out != ref.out) {
                    if (n_diff++ < 5) {
                        fprintf(stderr, "spec sim: seed %u drafter %d replay %d: output differs from the non-speculative one\n", seed, dt, replay);
                    }
                }
                if (!replay) {
                    n_cycles[dt] += g.n_cycles;
                    n_acc[dt]    += g.n_accepted;
                    if (dt == DRAFT_ADAPTIVE) {
                        for (int i = 0; i <= n_draft_long; ++i) {
                            n_len_adaptive[i] += g.n_len[i];
                        }
                    }
                }
            }
        }

        // PIT of the emitted tokens: u = F(t-) + V p(t) under the exact conditional distribution
        llama_sampler * rchain = make_chain(nullptr);
        std::mt19937 rng(seed);
        llama_token last = 0;
        for (int i = 0; i < 5; ++i) {
            last = m.id_of[rng() % n_tok];
            llama_sampler_accept(rchain, last);
        }
        std::mt19937 vrng(seed ^ 0x5bd1e995u);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        double u_prev = -1.0;
        for (const llama_token t : ref.out) {
            const auto p = exact_probs(m, rchain, last);
            const int ti = m.index(t);
            if (p[ti] <= 0.0) {
                fprintf(stderr, "spec sim: seed %u emitted a token with probability 0\n", seed);
                ok = false;
            }
            double F = 0.0;
            for (int v = 0; v < ti; ++v) {
                F += p[v];
            }
            const double u = std::min(std::max(F + ud(vrng)*p[ti], 0.0), 1.0 - 1e-12);
            n_bins[(int) (u*20)]++;
            n_u++;
            if (u_prev >= 0.0) {
                n_pair[(int) (u_prev*5)][(int) (u*5)]++;
                n_p++;
            }
            u_prev = u;
            llama_sampler_accept(rchain, t);
            last = t;
        }
        llama_sampler_free(rchain);
    }

    double chi2_u = 0.0, chi2_p = 0.0;
    for (int b = 0; b < 20; ++b) {
        const double e = n_u/20.0;
        chi2_u += (n_bins[b] - e)*(n_bins[b] - e)/e;
    }
    for (int a = 0; a < 5; ++a) {
        for (int b = 0; b < 5; ++b) {
            const double e = n_p/25.0;
            chi2_p += (n_pair[a][b] - e)*(n_pair[a][b] - e)/e;
        }
    }
    const double pv_u = chi2_pvalue(chi2_u, 19);
    const double pv_p = chi2_pvalue(chi2_p, 24);

    printf("speculative simulation: %d seeds x %d tokens, drafters coupled/noise/oracle/random/adaptive x replay off/on\n", n_seeds, n_gen);
    printf("  outputs different from the non-speculative run: %d\n", n_diff);
    printf("  accepted/cycle: coupled %.2f, noise %.2f, oracle %.2f, random %.2f, adaptive %.2f\n",
            (double) n_acc[DRAFT_COUPLED]/n_cycles[DRAFT_COUPLED], (double) n_acc[DRAFT_NOISE]/n_cycles[DRAFT_NOISE],
            (double) n_acc[DRAFT_ORACLE]/n_cycles[DRAFT_ORACLE], (double) n_acc[DRAFT_RANDOM]/n_cycles[DRAFT_RANDOM],
            (double) n_acc[DRAFT_ADAPTIVE]/n_cycles[DRAFT_ADAPTIVE]);
    printf("  adaptive draft lengths:");
    for (int i = 1; i <= n_draft_long; ++i) {
        printf(" %d:%" PRId64, i, n_len_adaptive[i]);
    }
    printf("\n");
    printf("  PIT: chi2 %.2f (dof 19, p-value %.3f), consecutive pairs chi2 %.2f (dof 24, p-value %.3f)\n", chi2_u, pv_u, chi2_p, pv_p);

    ok &= n_diff == 0 && pv_u > 1e-4 && pv_p > 1e-4;
    return ok;
}

} // namespace spec_sim

static bool test_mtp_padding() {
#ifndef GGML_USE_CPU
    printf("MTP padded-head confidence: skipped (CPU backend not linked)\n");
    return true;
#else
    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend);
    ggml_backend_cpu_set_n_threads(backend, 1);

    ggml_init_params params = { 2*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    llama_mtp_draft_head dh;
    llm_graph_input_mtp_draft inp(1, 1, 1, &dh);
    inp.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    inp.h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    inp.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    inp.rng = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);

    constexpr int n_rows = 1024;
    ggml_tensor * raw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rows, 1);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_rows);
    ggml_tensor * logits = inp.mask_logits(ctx, raw);
    ggml_tensor * prob = ggml_soft_max(ctx, logits);
    ggml_tensor * pick = ggml_argmax(ctx, logits);
    ggml_tensor * noisy = ggml_add_gumbel(ctx, logits, ids, inp.rng, 0);
    ggml_tensor * confidence = ggml_soft_max(ctx, noisy);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * output : {prob, pick, confidence}) {
        ggml_build_forward_expand(gf, output);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    std::vector<float> values(n_rows, 0.0f);
    values[1] = 1.0f;
    std::vector<float> token_ids(n_rows, 0.0f);
    token_ids[1] = 1.0f;
    ggml_backend_tensor_set(raw, values.data(), 0, values.size()*sizeof(float));
    ggml_backend_tensor_set(ids, token_ids.data(), 0, token_ids.size()*sizeof(float));

    llama_token token = 0;
    llama_pos pos = 0;
    float state = 0.0f;
    llama_ubatch ubatch = {};
    ubatch.n_tokens = 1;
    ubatch.token = &token;
    ubatch.pos = &pos;
    ubatch.embd_state = &state;

    bool ok = true;
    // Update valid rows without rebuilding the graph, including shrink after reset.
    for (int n_used : {2, 3, 2, n_rows}) {
        dh.n_used = n_used;
        for (float temp : {0.0f, 0.7f}) {
            dh.noise_temp = temp;
            inp.set_input(&ubatch);
            GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
            std::vector<float> p(n_rows), c(n_rows);
            int32_t selected = -1;
            ggml_backend_tensor_get(prob, p.data(), 0, p.size()*sizeof(float));
            ggml_backend_tensor_get(confidence, c.data(), 0, c.size()*sizeof(float));
            ggml_backend_tensor_get(pick, &selected, 0, sizeof(selected));
            const double expected = std::exp(1.0)/(std::exp(1.0) + n_used - 1);
            printf("  valid=%d temp=%.1f p[1]=%.9f expected=%.9f\n", n_used, temp, p[1], expected);
            ok &= selected == 1 && std::fabs(p[1] - expected) < 1e-6;
            double sum_p = 0.0, sum_c = 0.0;
            for (int i = 0; i < n_rows; ++i) {
                ok &= std::isfinite(p[i]) && std::isfinite(c[i]);
                if (i >= n_used) {
                    ok &= p[i] == 0.0f && c[i] == 0.0f;
                }
                if (temp == 0.0f) {
                    ok &= std::fabs(p[i] - c[i]) < 1e-6;
                }
                sum_p += p[i];
                sum_c += c[i];
            }
            ok &= std::fabs(sum_p - 1.0) < 1e-6 && std::fabs(sum_c - 1.0) < 1e-6;
        }
    }
    printf("MTP padded-head confidence and graph reuse: %s\n", ok ? "OK" : "FAILED");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
#endif
}

int main(int argc, char ** argv) {
    const int64_t n_samples = argc > 1 ? atoll(argv[1]) : 200000;

    bool ok = test_mtp_padding();

    if (argc > 1 && std::string(argv[1]) == "--mtp-padding-only") {
        return ok ? 0 : 1;
    }

    ok &= spec_sim::test((int) std::max<int64_t>(n_samples/100, 200));

    // spread 2: top-p cuts, spread 0.5: top-k cuts
    for (const float spread : { 2.0f, 0.5f }) {
        const auto base = make_logits(spread);
        for (const float temp : { 1.0f, 0.7f }) {
            ok &= test_distribution(base, spread, temp, n_samples);
        }
        ok &= test_greedy(base);
    }

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
