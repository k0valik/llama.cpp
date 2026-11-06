#!/usr/bin/env bash
# Run a trusted copy of this script outside the worktree it updates.
set -Eeuo pipefail
: "${SYNC_REPORT:?Set SYNC_REPORT to a path outside the worktree}"
: "${SYNC_UPSTREAM_URL:=https://github.com/ggml-org/llama.cpp.git}"
: "${SYNC_BASE:=port/qwen3.8-27b-cuda-v0.9.0}"
: "${SYNC_BRANCH:=sync/upstream-v090}"
: "${SYNC_UPSTREAM_BRANCH:=master}"
export GIT_TERMINAL_PROMPT=0

git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git config core.hooksPath /dev/null
git config rerere.enabled false
# Disable accidental network access through submodules, and never run a build.
git config submodule.recurse false

git fetch --no-tags origin "+refs/heads/$SYNC_BASE:refs/remotes/sync-base" "+refs/heads/$SYNC_BRANCH:refs/remotes/sync-candidate"
git fetch --no-tags "$SYNC_UPSTREAM_URL" "+refs/heads/$SYNC_UPSTREAM_BRANCH:refs/remotes/sync-upstream"
base=$(git rev-parse refs/remotes/sync-base)
candidate=$(git rev-parse refs/remotes/sync-candidate)
upstream=$(git rev-parse refs/remotes/sync-upstream)
printf '# Rolling upstream synchronization\n\n- Canonical: `%s`\n- Published candidate before run: `%s`\n- Upstream: `%s`\n\n' "$base" "$candidate" "$upstream" > "$SYNC_REPORT"
printf 'This is an unvalidated integration candidate, not a daily-driver promotion.\n\n' >> "$SYNC_REPORT"

git switch --detach "$candidate"
for source in sync-base sync-upstream; do
    if ! git -c core.hooksPath=/dev/null -c rerere.enabled=false merge --no-edit --no-ff "refs/remotes/$source"; then
        {
            printf '## Blocked: conflict merging `%s`\n\n' "$source"
            printf 'No candidate changes from this run were pushed. Conflicting paths:\n\n```\n'
            git diff --name-only --diff-filter=U
            printf '```\n\nResolve semantically in an isolated worktree and push a reviewed merge. Clean hunks also need review.\n'
        } >> "$SYNC_REPORT"
        exit 1
    fi
done
new=$(git rev-parse HEAD)
if [ "$new" != "$candidate" ]; then
    # A normal push is intentionally used: a concurrent writer must win, not be overwritten.
    if ! git push origin "HEAD:refs/heads/$SYNC_BRANCH"; then
        printf '## Blocked: push rejected\n\nThe remote may have changed; no force push was attempted. Retry after inspection.\n' >> "$SYNC_REPORT"
        exit 1
    fi
fi
printf '## Result\n\nPublished candidate: `%s`. No build or inference tests were run by this workflow.\n' "$new" >> "$SYNC_REPORT"
