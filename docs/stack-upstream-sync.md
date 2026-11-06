# Rolling upstream integration

The daily default branch remains `port/qwen3.8-27b-cuda-v0.9.0`. A daily
`stack-upstream-sync` workflow uses trusted code from that default branch to
prepare `sync/upstream-v090` and maintain its draft PR. It can also be run with
Actions → Stack upstream sync → Run workflow. Schedule times are UTC and GitHub
may delay or skip scheduled runs. Actions needs permission to create PRs.

The job fetches the canonical branch, the published candidate, and
`ggml-org/llama.cpp` master, then merges canonical followed by upstream in a
fresh checkout. It publishes only if both merges complete and a normal
(non-force) push succeeds. It does not run builds, tests, or scripts from the
merged tree. The worker script is copied out of the checkout before switching
branches. Runs are serialized; simultaneous human pushes are protected by Git's
non-fast-forward rejection.

The PR body and Actions summary are **generated and replaced** on every run.
They contain exact input and resulting commit IDs, or conflicting paths if a
merge fails. The existing remote candidate is not changed by a conflicting run;
its previous state may be stale. Inspect the workflow log if fetch or another
error occurs before the report is written. Do not treat an automatic clean merge
as reviewed, built, or numerically validated. The workflow is not a release or
promotion mechanism and never pushes canonical.

For conflicts, fetch the same source commits and merge them in a separate
worktree, inspect both conflict and clean changes, resolve semantically, and
push the resulting merge to the candidate branch without rewriting history.
Rerere is disabled in the automation and should not replace semantic review.
Validate CUDA, attention, MTP, recurrent state and inference as applicable
before intentionally promoting a candidate to the daily branch. For worker
regression tests run `python3 tests/test-stack-upstream-sync.py`; they use only
disposable local repositories.
