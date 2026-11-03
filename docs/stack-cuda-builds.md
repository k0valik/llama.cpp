# Stack CUDA build workflow

`.github/workflows/stack-cuda-build.yml` builds two source variants for Linux and Windows:

- `reference`: upstream commit `4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0` plus the original feveromo patch, checked by SHA-256.
- `candidate`: the exact commit on which the workflow was triggered. Check its commit and validation status before treating it as a completed port.

Each uses CUDA 13.1, `120a-real` for the RTX 5070 Ti, CUDA graphs, and a conservative AVX2 CPU baseline rather than the hosted runner's native CPU. Windows uses MSVC; Linux uses GCC. These settings make the two variants comparable on the same OS, but are not identical to the recipe's original Linux compiler and flags or the local Windows clang-cl script. Compare matching artifacts from the same workflow run. The artifacts include the CUDA runtime libraries, `BUILD-INFO.txt`, CMake cache, server/CLI/bench binaries and available test binaries. The coupled-sampling test is not registered by the reference or target for Windows shared-library builds; it is built on Linux when present. UI and OpenSSL are disabled.

## Trigger without changing the default branch

GitHub requires a workflow to exist on the default branch before the Actions UI or `workflow_dispatch` can run it. Until this workflow is added there, push a **new** tag at the desired commit:

```sh
git switch port/feveromo-v0.8.0
git pull --ff-only
git tag stack-build/2026-11-03-01 HEAD
git push origin stack-build/2026-11-03-01
```

Use a unique tag on each run; never retarget an existing tag. The tag must point to a commit containing the workflow. A normal tag push launches all four jobs; regular branch pushes do not. To retry only the two candidate jobs, use a tag starting with `stack-build/candidate-only-`; the reference jobs are omitted. After the workflow is on the default branch, it can also be launched from the Actions UI or with:

```sh
gh workflow run stack-cuda-build.yml --repo k0valik/llama.cpp --ref port/feveromo-v0.8.0
```

## Download and test

Open the run in the repository's Actions tab and download the four named artifacts, or use `gh run download RUN_ID --repo k0valik/llama.cpp`. Artifacts expire after seven days. Extract each archive into a separate directory. Do not mix DLLs, shared libraries or binaries between variants.

Hosted standard runners compile only; they do not have the target GPU or model. A successful build is not evidence of numerical correctness, performance, or successful deployment. Run the included tests and matched-model comparisons on the target machine, saving logs and build metadata. Windows needs a compatible NVIDIA driver and the Microsoft Visual C++ runtime; Linux needs a compatible NVIDIA driver and system libraries appropriate to the build image.

The workflow is initially unverified. A failed configure, compile or package step produces no artifact for that job. Consult its run logs before using anything that did build. Standard GitHub-hosted runners are free for public repositories under GitHub's billing policy; storage and billing rules can change.
