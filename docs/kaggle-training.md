# Pasture Studies Kaggle training guide

This guide and `notebooks/pasture_studies_kaggle.ipynb` provide a reproducible Kaggle entry point for the local PyTorch trainer. The NumPy policy/trainer remain the baseline. The MVP1 failure report and its acceptance thresholds are preserved. This is a synthetic simulation experiment; it is not biologically validated and is not production-husbandry guidance.

## What this run can and cannot establish

- Compare trajectory REINFORCE and vanilla A2C on CPU with matching seeds/configuration.
- Run A2C with an explicitly selected device and persist metrics/checkpoints for later analysis.
- A CUDA device being visible is **not** proof of acceleration. Report timed evidence, device count, runtime, config and completion status; distinguish fixed-work device measurements from dynamic end-to-end training. Do not label a CPU fallback as a GPU result.
- The trainer supports two-process NCCL DDP when launched with the documented `torchrun` command below. A T4x2 result still requires confirming two visible devices, completed distributed updates, and the plan's performance/parity gates; do not infer acceleration from DDP support.
- Higher reward is not a cognition gate. Report action distributions, entropy, learning curves, and the trainer's paired memory/perception influence metrics. Do not infer biological validity from any result.

## Kaggle setup and run

1. Open `notebooks/pasture_studies_kaggle.ipynb` in Kaggle (or create a notebook and import this file from the public `santleme/pasture-studies` GitHub repository).
2. In notebook settings choose Python, enable Internet only for GitHub/pip installation, and select **GPU: T4 x2** if it is offered. Availability/quota can depend on account and demand. Kaggle's documented T4x2 profile is two T4 devices, 4 CPU cores, 29 GB RAM and up to 12 hours; actual allocation and quota should be checked in the current session. Saved notebook outputs are limited; keep artifacts comfortably below the documented 20 GB output allowance.
3. Use the notebook's `PASTURE_PIPELINE_COMMIT`, pinned to `74f5ccd5b6dd785b2e424d29302c15bcec73bbc1`. The setup fetches that exact commit, checks out detached HEAD, and aborts on any mismatch. Do not replace it with a moving branch for confirmatory work. The final notebook-pin commit is separate because a commit cannot contain its own SHA. Optionally set `PASTURE_NOTEBOOK_REVISION` to the imported Kaggle notebook revision and record it in run provenance.
4. Run all setup cells. The optional training dependencies are installed from `requirements-training.txt`; the notebook runs `pip check` and prints the trainer's `--help` output. Inspect the resolved PyTorch version, visible CUDA count and device names.
5. Generate the outcome-blind preregistration from the pinned checkout before training. The notebook records its SHA-256 and reads `training_root_seeds` and episode count directly from it. Then run the disposable `--smoke` cell; its checkpoint is not a formal training/evaluation arm.
6. The notebook's matched comparison runs CPU REINFORCE and CPU A2C with identical seed IDs and episode counts. The CUDA arm is A2C only. When two GPUs are visible, it runs T4x1 (single visible device) and T4x2 (two-rank DDP) dynamic conditions, then the separate fixed-corpus compute benchmark. If CUDA/T4x2 is unavailable, CPU comparison remains possible; the unavailable GPU results stay **not measured**.
7. The run ID defaults to `mvp1-run`; artifact paths are stable at `/kaggle/working/pasture-training-artifacts-mvp1-run`. Rerunning with the same ID automatically resumes existing per-arm checkpoints after verifying the source commit. Use a new `PASTURE_RUN_ID` for a genuinely separate experiment. For an interruption, the full-run cell detects `LATEST`; the explicit resume example can recover one CPU arm. Do not mix incompatible configs. Recovery across a Kaggle session reset is possible only if Kaggle retained the directory or you restore the downloaded checkpoint artifacts to that same run path.
8. Run the export cell, then use Kaggle **Save Version → Save & Run All**. Wait for the version run to finish before leaving the session. `/kaggle/working` is temporary until outputs are attached to a saved notebook version; download the archive if you need guaranteed recovery outside the current session.
9. In the saved notebook's **Output** panel, download `export/pasture-training-artifacts.zip` and `export/artifact_manifest.json` (or download the relevant checkpoint files individually). Keep the source commit, run config, metrics, and manifest together. Verify downloaded hashes locally before inference.

Kaggle documentation: [Notebooks](https://www.kaggle.com/docs/notebooks), [Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage). GPU use may be limited by account quota/availability; these published resource ceilings do not guarantee allocation.

## Direct command-line workflow

These commands can run in a Kaggle notebook cell (prefix shell commands with `!`) or a Linux shell. The pipeline source commit is `74f5ccd5b6dd785b2e424d29302c15bcec73bbc1`; the published notebook pin commit adds the strict SHA and this final publication documentation. The notebook itself fetches the source commit above. No GitHub write token is needed; the pipeline must not push artifacts back to the public repository. Generate the preregistration only after all source-bound files are final; it reads no outcomes, verifies hashes, and records the manifest SHA-256. Use that generated file for training and validation evaluation.

```bash
# Source checkout and provenance
git clone https://github.com/santleme/pasture-studies.git
cd pasture-studies
git checkout --detach 74f5ccd5b6dd785b2e424d29302c15bcec73bbc1
git rev-parse HEAD | tee source_commit.txt

# Optional local venv; Kaggle's notebook kernel can use pip directly
python -m pip install -r requirements-training.txt
python -m pip check
python -m validation.train_torch_mvp1 --help

# Freeze and bind the outcome-blind protocol BEFORE outcome collection
export PREREG=/kaggle/working/pasture-training-artifacts-mvp1-run/preregistration.json
python validation/preregister_torch_mvp1.py --output "$PREREG"
python - <<'PY'
import hashlib, json, os
p=os.environ["PREREG"]
m=json.load(open(p))
assert m["status"] == "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0"
print("manifest_sha256:", m["manifest_sha256"])
print("training_root_seeds:", m["seed_splits"]["training_root_seeds"])
print("episodes_per_train_seed:", m["training_config"]["episodes_per_train_seed"])

PY

# Required short pipeline check before a long run
python -m validation.train_torch_mvp1 \
  --smoke --algorithm both --device cpu --seed 71001 \
  --output /kaggle/working/pasture-training-artifacts-mvp1-run/smoke \
  --metrics /kaggle/working/pasture-training-artifacts-mvp1-run/smoke/metrics.jsonl \
  --checkpoint-every 1
```

Matched CPU algorithm comparison (repeat for each `training_root_seeds` entry from `$PREREG`; current generated roots are 71001–71010. Do not substitute hand-written roots. The preregistered episode count is 278):

```bash
SEED=71001 # example only; iterate exact seed_splits.training_root_seeds from preregistration
EPISODES=278 # read training_config.episodes_per_train_seed from preregistration
ROOT=/kaggle/working/pasture-training-artifacts-mvp1-run/matched_cpu
for ALG in reinforce a2c; do
  python -m validation.train_torch_mvp1 \
    --mode train --algorithm "$ALG" --device cpu \
    --seed "$SEED" --episodes "$EPISODES" \
    --output "$ROOT/$ALG/seed-$SEED" \
    --metrics "$ROOT/$ALG/seed-$SEED/metrics.jsonl" \
    --checkpoint-every 16
done
```

A2C T4x1 (single visible device) and T4x2 (two-rank DDP) conditions:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))
PY

SEED=71001 # example only; use preregistration training_root_seeds
CUDA_VISIBLE_DEVICES=0 python -m validation.train_torch_mvp1 \
  --mode train --algorithm a2c --device cuda \
  --seed "$SEED" --episodes 278 \
  --output "/kaggle/working/pasture-training-artifacts-mvp1-run/a2c_accelerator/t4x1/seed-$SEED" \
  --metrics "/kaggle/working/pasture-training-artifacts-mvp1-run/a2c_accelerator/t4x1/seed-$SEED/metrics.jsonl" \
  --checkpoint-every 16
```

Resume a run with the exact same arguments/config except add `--resume latest` (or `--resume /path/to/checkpoint-generation`):

```bash
python -m validation.train_torch_mvp1 \
  --mode train --algorithm a2c --device cpu --seed 71001 --episodes 278 \
  --output /kaggle/working/pasture-training-artifacts-mvp1-run/matched_cpu/a2c/seed-71001 \
  --metrics /kaggle/working/pasture-training-artifacts-mvp1-run/matched_cpu/a2c/seed-71001/metrics.jsonl \
  --checkpoint-every 16 --resume latest
```

CLI flags and checkpoint layout are versioned with the checked-out source; always inspect that revision's `--help` output. The Kaggle notebook uses the same CLI. This trainer supports `torchrun` with two ranks, NCCL, complete episode shards (16-episode batches split 8/8 and six-episode tails split 3/3), deterministic world IDs, and rank-zero checkpointing. DDP does not shard inputs automatically; this explicit engine protocol is required. Recovery is whole-job restart from a committed global-update boundary; elastic recovery of a single failed rank while other ranks continue is not implemented or claimed.

## Retrieve and verify artifacts locally

Unzip the Kaggle download without overwriting an existing experiment:

```bash
mkdir -p artifacts/kaggle-run-71001
unzip pasture-training-artifacts.zip -d artifacts/kaggle-run-71001
```

The notebook emits `artifact_manifest.json` with source commit, runtime/device information, per-file size and SHA-256. Verify each listed file before use (from repository root):

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('artifacts/kaggle-run-71001')
manifest = json.loads((root / 'export/artifact_manifest.json').read_text())
for item in manifest['files']:
    p = root / item['path']
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != item['sha256']:
        raise SystemExit(f"HASH MISMATCH: {p}")
print(f"Verified {len(manifest['files'])} artifacts from {manifest['repo_url']} @ {manifest['commit']}")
PY
```

For local cattle-simulation inference, use the **NumPy inference export** (`.npz`) and its matching JSON sidecar from the same checkpoint generation; the optimizer/RNG resume state is not the inference file. Keep both files adjacent. Example from repository root:

```bash
python - <<'PY'
from farm_sim.policy import load_checkpoint
policy = load_checkpoint('artifacts/kaggle-run-71001/a2c_accelerator/t4x2/seed-71001/a2c/torch-a2c-seed-71001.npz')
print('Loaded NumPy policy:', type(policy).__name__)
PY
```

Find the concrete NPZ name by listing the downloaded generation directory. Do not rename or pair an NPZ with a sidecar from another generation. `load_checkpoint` validates the existing NumPy policy schema; this does not establish cross-version behavioral parity by itself. Preserve the matching generation manifest and report any schema/version rejection rather than bypassing it.

## Reproducibility and interpretation

Record at minimum: full repository commit; package versions; device model/count; algorithm; all CLI arguments; per-seed output; checkpoint generation/hash; training/evaluation seed split; and whether resumption occurred. Seed-controlled runs are reproducible only within documented software/device constraints; different PyTorch/CUDA/hardware versions may not be bitwise identical. Do not compare unlike episode counts or task conditions as if they were a matched algorithm experiment.

Keep GPU performance reports distinct:

1. **Network/fixed-work compute**: same frozen training-only inputs and optimizer work across CPU/T4 conditions; CUDA synchronization/events and repeated paired measurements are required.
2. **Dynamic end-to-end pipeline**: full environment + policy run under matching configuration, with trajectories and simulator work counters reported. Same seed IDs do not imply identical trajectories after floating-point/policy divergence.

A GPU PASS requires the approved plan's evidence and thresholds; device visibility, a successful run, or a faster single seed is insufficient. Preserve MVP1's original failed cognition/learning gates and report failed/inconclusive outcomes without tuning thresholds after observing results. The public checkout carries the approved plan/review SHA-256 pins in `validation/kaggle-mvp1/approved-plan-hashes.json` and copies the exact acceptance gates into `validation/kaggle-mvp1/preregistration.json`; the original approval files remain local OMX artifacts and are intentionally not distributed.

Official PyTorch references: [Reproducibility (2.14)](https://docs.pytorch.org/docs/2.14/notes/randomness.html), [CUDA semantics (2.14)](https://docs.pytorch.org/docs/2.14/notes/cuda.html), [DDP API (2.14)](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html), [Saving/loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). CUDA operations are asynchronous; benchmark timing must synchronize or use CUDA events. DDP gradient synchronization alone does not implement correct input sharding.

For a T4x2 run on Kaggle (two visible T4 GPUs), use the supported DDP launcher. `episodes=278` yields 17 full 16-episode batches and a six-episode tail, which divides evenly 8/8 and 3/3 across the two ranks:

```bash
SEED=71001
OUT="/kaggle/working/pasture-training-artifacts-mvp1-run/a2c_accelerator/t4x2/seed-$SEED"
mkdir -p "$OUT"
torchrun --standalone --nproc_per_node=2 -m validation.train_torch_mvp1 \
  --mode train --algorithm a2c --device cuda \
  --seed "$SEED" --episodes 278 --output "$OUT" \
  --metrics "$OUT/metrics.jsonl" --checkpoint-every 16
```

For the paired single-T4 condition when Kaggle exposes T4x2, hide GPU 1 so this run is an actual single-device condition:

```bash
CUDA_VISIBLE_DEVICES=0 python -m validation.train_torch_mvp1 \
  --mode train --algorithm a2c --device cuda \
  --seed 71001 --episodes 278 \
  --output /kaggle/working/pasture-training-artifacts-mvp1-run/a2c_accelerator/t4x1/seed-71001 \
  --metrics /kaggle/working/pasture-training-artifacts-mvp1-run/a2c_accelerator/t4x1/seed-71001/metrics.jsonl \
  --checkpoint-every 16
```

## Fixed-corpus compute benchmark (separate result from dynamic training)

The notebook runs the full `kaggle-a2c-fixed-corpus-v1` protocol separately from closed-loop training. This benchmark first creates one frozen, training-only corpus and then reuses its exact hashes on CPU, one visible T4, and two-rank T4x2. It performs five full repeats by default (10 partitions × 278 complete episodes; 180 optimizer updates per repeat). Corpus generation time is outside timed replay. The corpus is intentionally stale/off-policy during replay; the resulting weights are compute fixtures, never policy-learning evidence or inference candidates.

Exact CLI commands (run from the repository root after installation):

```bash
FIXED=/kaggle/working/pasture-training-artifacts-mvp1-run/fixed_corpus_benchmark
mkdir -p "$FIXED"
python -m validation.benchmark_torch_fixed_corpus generate-corpus \
  --seed 51337 --output "$FIXED/kaggle-a2c-fixed-corpus-v1.npz"

# CPU condition
python -m validation.benchmark_torch_fixed_corpus benchmark \
  --corpus "$FIXED/kaggle-a2c-fixed-corpus-v1.npz" --device cpu \
  --repeats 5 --seed 51337 --output "$FIXED/cpu.json"

# T4x1 condition even if Kaggle exposes two GPUs
CUDA_VISIBLE_DEVICES=0 python -m validation.benchmark_torch_fixed_corpus benchmark \
  --corpus "$FIXED/kaggle-a2c-fixed-corpus-v1.npz" --device cuda \
  --repeats 5 --seed 51337 --output "$FIXED/t4x1.json"

# T4x2 condition: two-rank NCCL DDP, both T4 GPUs visible
torchrun --standalone --nproc_per_node=2 -m validation.benchmark_torch_fixed_corpus benchmark \
  --corpus "$FIXED/kaggle-a2c-fixed-corpus-v1.npz" --device cuda \
  --repeats 5 --seed 51337 --output "$FIXED/t4x2.json"

# Paired elapsed-time ratios: first report is numerator, second denominator
python -m validation.benchmark_torch_fixed_corpus compare \
  --first "$FIXED/cpu.json" --second "$FIXED/t4x1.json" --output "$FIXED/cpu-vs-t4x1.json"
python -m validation.benchmark_torch_fixed_corpus compare \
  --first "$FIXED/cpu.json" --second "$FIXED/t4x2.json" --output "$FIXED/cpu-vs-t4x2.json"
python -m validation.benchmark_torch_fixed_corpus compare \
  --first "$FIXED/t4x1.json" --second "$FIXED/t4x2.json" --output "$FIXED/t4x1-vs-t4x2.json"
```

A deliberately reduced corpus is available only for pipeline smoke checks: pass `--smoke` to both `generate-corpus` and `benchmark`; it is explicitly marked and cannot count toward acceptance. Full fixed-work interpretation requires all five paired repeats, complete full-workload reports, identical corpus/config/source hashes, and the DDP global-mean gradient, resume/recovery and provenance tests. Apply the frozen thresholds copied into `validation/kaggle-mvp1/preregistration.json`: GPU-vs-best-CPU fixed-work throughput 95% CI lower bound ≥2.0 and ≥30% elapsed-time reduction; T4x2 additionally ≥1.20 vs. T4x1. Threshold miss after complete valid timing is a fixed-work **FAIL**. Missing/reduced/mismatched evidence is **INCONCLUSIVE/UNMEASURED**. Even if timing criteria clear, a final PASS requires the independent acceptance tests/review; device visibility and MLP microbenchmarks alone are never a PASS. The benchmark's `interpretation` field is descriptive-only by design.

## Manifest-defined validation policy and cognition evaluation

The notebook invokes `validation.evaluate_torch_mvp1` with the same source-bound preregistration generated before training and `--split validation`. The manifest freezes 30 validation world seeds per scenario family (starter, resource-scarce relocated, wide, unseen corridor); each checkpoint is evaluated on the same worlds with trained, same-seed untrained, and heuristic policies. Reports include reward components, action counts/frequencies/entropy, invalid actions, navigation reasons, final physiological/resource/position states, and paired memory/perception interventions/ablations with input hashes/shams. Keep reports per training seed and checkpoint rather than pooling away variation.

Example for a trained A2C checkpoint, using a training root read from the manifest:

```bash
TRAIN_SEED=71001 # example only: iterate seed_splits.training_root_seeds from $PREREG
GEN=/kaggle/working/pasture-training-artifacts-mvp1-run/matched_cpu/a2c/seed-${TRAIN_SEED}/a2c
python -m validation.evaluate_torch_mvp1 \
  --generation "$GEN" --algorithm a2c --seed "$TRAIN_SEED" \
  --seed-manifest "$PREREG" --split validation --device cpu \
  --output "/kaggle/working/pasture-training-artifacts-mvp1-run/heldout_evaluation/a2c/train-seed-${TRAIN_SEED}.json"
```

This is **validation/diagnostic evidence**, not a locked-test acceptance result. The current preregistration status is `PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0`; `delta0` has not yet been established by the precision-only pilot. `--split locked_test` is intentionally not runnable from this manifest. A separate pre-outcome lock artifact must first bind the pilot-derived `delta0`, source hashes, selected checkpoint hashes, and analysis/split status; only then can any locked outcomes be accessed. Further, current evaluator output does not itself implement the complete multi-checkpoint clustered bootstrap/Holm cognition analysis. Keep the historical failure report and thresholds unchanged; no validation report can establish biological validity.

## Dynamic pipeline benefit: separate, conservative verdict

The notebook measures each complete dynamic training invocation from subprocess start through committed checkpoints and final evaluation, and retains the per-seed wall-time records under each run's `run_timing.json`. Compare CPU A2C, T4x1 A2C and T4x2 A2C only as a practical same-config pipeline comparison. Use the plan's independent gate: at least five paired complete runs; 95% CI lower speedup ≥2× and elapsed reduction ≥30% vs. CPU (T4x2 additionally ≥1.20× vs. T4x1); exact nominal update/transition counts; action disagreement ≤0.5%; and each named cumulative simulator-work counter within 1.0%. A dynamic run cannot pass the fixed-work GPU-compute gate. The present CLI timing records do not include the required cross-device action/trajectory and simulator-work divergence comparison, so the notebook correctly reports dynamic pipeline benefit as **INCONCLUSIVE** even when elapsed times are available. Report raw timings descriptively; do not call a faster run a speedup claim.

The generated preregistration supplies disjoint train/validation/pilot/locked splits, but its current status remains pending the precision-only pilot: `delta0` and a pre-outcome lock artifact binding source/checkpoint hashes are not yet available. The evaluator's validation outputs and first-decision memory/perception probes are diagnostic only; they do not implement the full multi-checkpoint clustered bootstrap/Holm procedure or satisfy the locked acceptance gates. The four synthetic scenario families do not establish biological validity.
