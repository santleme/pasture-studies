# Pasture Studies Kaggle Cognition Pipeline — Verification Report

**Result: PASS for locally verifiable implementation gates; INCONCLUSIVE / NOT RUN for Kaggle hardware and learned-cognition acceptance.** This report covers MVP 1 remediation only. No biological or husbandry validation is claimed, and MVP 2 was not started.

## Delivered

- Optional PyTorch CPU/CUDA neural training while retaining the NumPy policy/training baseline: `farm_sim/torch_training.py`, `validation/train_torch_mvp1.py`.
- Matched trajectory REINFORCE and vanilla A2C with deterministic seed derivation, batched multi-world inference, training curves/action and entropy metrics, reward and cognition-influence probes, invalid-output accounting, and NPZ + JSON local-inference export.
- Hash-checked resumable checkpoint generations with optimizer/training state, safe weights-only load, configuration validation, and recovery of complete generations left between directory rename and `LATEST` update: `farm_sim/torch_checkpoint.py`.
- Fixed-work benchmark and evaluator including held-out validation scenarios, policy comparisons, paired perception/memory interventions and ablations: `validation/benchmark_torch_fixed_corpus.py`, `validation/evaluate_torch_mvp1.py`.
- A source-bound, outcome-blind split protocol and tracked hash attestation for the locally approved plan/reviews: `validation/kaggle-mvp1/preregistration.json`, `validation/kaggle-mvp1/approved-plan-hashes.json`.
- Kaggle notebook and exact run/recovery/download/local-load instructions: `notebooks/pasture_studies_kaggle.ipynb`, `docs/kaggle-training.md`.

## Verification evidence

### Final cleanup/review gate

A bounded anti-slop audit covered the changed training, checkpoint, evaluator, preregistration, simulation, tests, notebook, and guide. Behavior was locked by the regression suite before the last cleanup/recovery pass. No dead compatibility branches, masking fallbacks, redundant abstractions, or unrelated files were identified that justified another code edit; the checkpoint fail-safe paths are explicit and failure-injection tested. UI cleanup is not applicable. Independent architecture review returned **PASS** with GPU/Kaggle execution as an explicit external caveat; independent code review returned **APPROVE** with zero findings.

### Software and reproducibility

- Full local regression suite: **121 tests run; 120 passed, 1 skipped**. The skip is the CUDA-only export-parity test because no CUDA device is available locally.
- Targeted checkpoint suite after recovery hardening: **12/12 passed**. Includes an injected interruption after generation rename but before pointer update; resume selected the valid orphan. Additional adversarial cases prove incompatible-config and invalid-cursor orphans do not advance `LATEST`.
- Preregistration suite: **11/11 passed**, including clean-checkout behavior with `.omx` approval files unavailable and a recomputed-manifest-hash train-world/locked-seed collision attack.
- Vectorized world/collector parity: exercised **100 seeds × 72 ticks** in tests. Decision trajectories, resource arbitration, events, and final physical state match the scalar simulator path; feature rows from a stale world snapshot are rejected before commit.
- DDP engine check ran locally with two CPU/Gloo ranks; gradient/optimizer parity against the global-sample-mean reference passed. This is not evidence of GPU/NCCL performance or rank-failure tolerance.
- Training smoke command ran on CPU with PyTorch `2.14.0+cpu`; both REINFORCE and A2C completed two updates and emitted resumable checkpoints, evaluation JSON, and NumPy inference exports. CUDA availability was false.
- Notebook JSON schema and all 32 code cells parsed/compiled successfully.
- `git diff --check` passed at final verification.

Reproduce the local smoke check from repository root:

```bash
python -m validation.train_torch_mvp1 \
  --smoke --algorithm both --device cpu --seed 71001 \
  --output /tmp/pasture-smoke --metrics /tmp/pasture-smoke/metrics.jsonl \
  --checkpoint-every 1
```

### Local CPU profile (descriptive only)

`validation/PROFILE_TORCH_PIPELINE_LOCAL.json` is regenerated with current profile/simulation sources. It reports a CPU-only comparison on three fixed seeds, 72 ticks per episode, five animals (1,080 decisions/repeat), three timing repeats:

| Collector | Median elapsed | p95 | Decisions/s at median |
|---|---:|---:|---:|
| Scalar per-animal inference | 0.22344 s | 0.22485 s | 4,834 |
| Batched multi-world inference | 0.15969 s | 0.15971 s | 6,763 |

The measured scalar-to-batched ratio is about **1.40×** for this local CPU workload. Paired outputs had exact observations/actions/rewards/uniforms/events/final physical state; maximum probability difference was `8.94e-8` under the declared tolerance. This profile is not a GPU benchmark, a general speedup guarantee, a training-quality result, or an acceptance gate.

### Outcome-blind manifest and frozen historical evidence

- Current pre-registration manifest SHA-256 (canonical manifest field): `484d57b94b288f903766870767e547081c5f2df70dec5fb8dce92c8e31edc270`.
- Verification confirms `PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0`, **3,390** new IDs disjoint from **320** historical IDs, and **zero** overlap. It contains 30 locked test seeds per each of four scenario families. No locked outcomes were read.
- The precision-only pilot has not produced a frozen `delta0`; utility and cognition confirmatory decisions remain gated. The historical failure report remains unchanged: `validation/MVP1_REMEDIATION_REPORT.md` SHA-256 `c11b4a04910045e9278ba703223d303c9f4c22f2536f0e0423f2bba5571166ee`.
- Approved thresholds are copied into the manifest and were not relaxed. The tracked approval-hash attestation is necessary because `.omx/` is intentionally absent from public clones.

## Acceptance gates

| Gate | Verdict | Evidence / reason |
|---|---|---|
| Existing NumPy baseline and simulator contract preserved | **PASS** | Full regression suite and scalar-vs-batched exact trajectory tests pass. |
| PyTorch CPU implementation; REINFORCE and A2C | **PASS (software)** | Oracle, loss/gradient, training, smoke, export, and deterministic-seed tests pass. Full training arms not executed. |
| Resume and interrupted checkpoint recovery | **PASS (local process interruption)** | Exact interrupted-vs-uninterrupted test plus injected rename/`LATEST` window and invalid-orphan tests. Whole-job recovery only; no elastic failed-rank recovery. |
| Reproducible, source-bound protocol and held-out split integrity | **PASS (protocol)** | Manifest verified against current source and approval hashes; splits audited disjoint. Locked outcomes remain sealed. |
| Vectorized batch semantics | **PASS** | 100-seed parity tests and fresh CPU profile agree within the numerical contract. |
| Local CPU/Gloo distributed gradient semantics | **PASS (software only)** | Two-rank Gloo reference parity checks pass. |
| Kaggle notebook/setup/artifact workflow | **PASS (static/local checks)** | 32 code cells compile; commands and pinning, GPU detection, run, resume, export, and local inference steps documented. No live Kaggle execution. |
| T4x1/T4x2 benchmark and GPU acceleration threshold | **NOT MEASURED / INCONCLUSIVE** | No local CUDA; Kaggle T4 hardware/repeats were not run. Do not claim GPU acceleration. |
| Full 10-seed matched learning comparison/generalization | **NOT RUN / INCONCLUSIVE** | Smoke runs are pipeline checks only. No checkpoint is promoted and no learned-policy utility claim is made. |
| Locked cognition causal gates / frozen `delta0` | **NOT RUN / INCONCLUSIVE** | The precision pilot and pre-outcome lock are pending; locked evaluation is blocked by design. Existing paired intervention/ablation harness is software-tested only. |
| Biological validity / production husbandry | **NOT VALIDATED; OUT OF SCOPE** | Simulator/reward/task are synthetic and not validated against cattle biology or husbandry evidence. |

## Exact Kaggle execution and local retrieval

Use `docs/kaggle-training.md` for the complete, copyable instructions. In brief: clone `https://github.com/santleme/pasture-studies.git`, check out an immutable reviewed commit, install `requirements-training.txt`, generate and verify the outcome-blind manifest, run smoke, then execute the exact preregistered CPU REINFORCE/A2C arms and CUDA A2C T4x1/T4x2 conditions. Run the fixed-corpus benchmark separately. Save a Kaggle notebook version and download `export/pasture-training-artifacts.zip` plus its `artifact_manifest.json`; verify file hashes locally. Use the matching checkpoint generation's `.npz` and JSON sidecar with `farm_sim.policy.load_checkpoint(...)` for local browser simulation inference. Do not treat optimizer-state checkpoints as inference files.

No commit/push was performed in this task. A public Kaggle run requires this work, including the tracked approval hash attestation, to be present in the immutable commit selected in the notebook. Kaggle/T4 measurements and the locked learning/cognition study remain the explicit external follow-up gates.
