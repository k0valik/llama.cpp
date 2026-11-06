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

# Show both outstanding upstream work and files changed independently by the fork.
# Overlap signals review risk, not an actual conflict or a semantic diagnosis.
ancestor=$(git merge-base "$candidate" "$upstream")
count=$(git rev-list --count "$ancestor..$upstream")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
{ git diff --name-only "$ancestor" "$candidate"; git diff --name-only "$ancestor" "$base"; } | LC_ALL=C sort -u > "$tmp/candidate-paths"
git diff --name-only "$ancestor" "$upstream" | LC_ALL=C sort > "$tmp/upstream-paths"
LC_ALL=C comm -12 "$tmp/candidate-paths" "$tmp/upstream-paths" > "$tmp/overlap"
overlap_count=$(wc -l < "$tmp/overlap")
{
    printf '## Upstream work since common ancestor\n\n'
    printf 'Common ancestor: `%s`. Upstream commits: **%s** (showing at most 60).\n\n```text\n' "$ancestor" "$count"
    git log --no-merges -60 --format='%h %s' "$ancestor..$upstream"
    printf '```\n\n## Files changed on both sides\n\n'
    printf '**%s** overlapping paths (showing at most 80). Even clean merges need semantic review.\n\n```text\n' "$overlap_count"
    awk 'NR <= 80 { print }' "$tmp/overlap"
    printf '```\n\n'
} >> "$SYNC_REPORT"

git switch --detach "$candidate"
if ! git merge --no-edit --no-ff "$base"; then
    {
        printf '## Blocked: conflict merging canonical `%s`\n\n' "$base"
        printf 'Nothing was pushed. Conflicting paths:\n\n```text\n'
        git diff --name-only --diff-filter=U
        printf '```\n'
    } >> "$SYNC_REPORT"
    exit 1
fi

# First-parent order preserves upstream ancestry; never skip a blocked commit.
# A merge commit here may bring its complete side branch along with it.
git rev-list --first-parent --reverse "$upstream" --not HEAD > "$tmp/commits"
applied=0
last=''
blocked=''
while IFS= read -r commit; do
    [ -n "$commit" ] || continue
    if ! git merge --no-edit --no-ff "$commit"; then
        blocked=$commit
        git diff --name-only --diff-filter=U > "$tmp/conflicts"
        # Unexpected errors must not be mistaken for a resolvable merge conflict.
        if [ ! -s "$tmp/conflicts" ]; then
            printf '## Blocked: merge failed without conflict paths for `%s`. Nothing was pushed; see the log.\n' "$commit" >> "$SYNC_REPORT"
            exit 1
        fi
        git merge --abort
        break
    fi
    applied=$((applied + 1))
    last=$commit
done < "$tmp/commits"

new=$(git rev-parse HEAD)
# A conflict at the first upstream commit does not publish a canonical-only merge.
if [ "$new" != "$candidate" ] && { [ -z "$blocked" ] || [ "$applied" -gt 0 ]; }; then
    # Never overwrite a concurrent human or automation update.
    if ! git push origin "HEAD:refs/heads/$SYNC_BRANCH"; then
        printf '## Blocked: push rejected\n\nNo force push was attempted; inspect the remote before retrying.\n' >> "$SYNC_REPORT"
        exit 1
    fi
    published=$new
else
    published=$candidate
fi
{
    printf '## Published candidate\n\n- Candidate: `%s`\n- First-parent upstream commits merged this run: **%s**\n' "$published" "$applied"
    if [ -n "$last" ]; then printf -- '- Last upstream commit merged: `%s`\n' "$last"; fi
    printf '\nThe PR diff is an unvalidated integration candidate; no build or inference tests were run.\n\n'
} >> "$SYNC_REPORT"
if [ -n "$blocked" ]; then
    {
        printf '## Blocked: conflict at upstream commit `%s`\n\n' "$blocked"
        printf 'Subject: %s\n\nConflicting paths:\n\n```text\n' "$(git show -s --format=%s "$blocked")"
        cat "$tmp/conflicts"
        printf '```\n\nNo later upstream commits were attempted. Resolve semantically and push a reviewed merge; clean hunks need review too.\n'
    } >> "$SYNC_REPORT"
    exit 1
fi
printf '## Result: mechanically clean, review required\n' >> "$SYNC_REPORT"
