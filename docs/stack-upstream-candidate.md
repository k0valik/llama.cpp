# Rolling upstream integration candidate

This branch is a review candidate, not the daily engine. Its rolling pull request
tracks merges from `ggml-org/llama.cpp` master into the canonical stack branch.
An initial marker commit makes the PR available even when the first upstream merge
has conflicts. Conflict reports describe what was attempted; a failed merge is not
published. Review clean changes as well as conflicts, and validate CUDA inference,
MTP and state handling before promoting anything into the daily driver.
