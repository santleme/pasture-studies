# Pasture Studies — Exploratory A2C Kaggle Results

**Status: exploratory evidence only. No candidate passed the requested combined behavior gate.**

This report records two real Kaggle notebook executions on `pascalcoimbra/pasture-studies` (the existing private notebook), v12 and v13. Both executions performed training and exported checkpoints, but both notebook jobs terminated with `ERROR` in post-training evaluation guards. The downloaded checkpoints and exports were independently re-hashed, restored, and evaluated locally. Do not interpret this as a successful `COMPLETE` notebook execution or as a formal scientific validation.

## Scope and integrity boundaries

- Original A2C/reward condition retained as control. The preregistered/formal protocol, thresholds, and historical failure records were not modified.
- Conditions A–D: three independent starts each, seeds `9483001–9483003`, 512 training episodes/model (12 training jobs).
- Follow-up E–F: one bounded, exploratory iteration with seeds `9483004–9483006`, 512 episodes/model (6 jobs). This used a stronger optional state-aware shaping scale after A–D failed to produce forage successes. Total: **18/24 authorized training jobs**; no additional jobs are recommended on this evidence alone.
- Evaluation training seeds and evaluation worlds were disjoint. Every condition was assessed on the same 12 generated worlds (3 each: `starter`, `resource_scarce_relocated`, `wide`, `unseen_corridor`), with 72 ticks/world. The evaluator produced 36 trained-policy episodes per condition (3 checkpoints × 12 worlds) and corresponding untrained/heuristic controls.
- Shared evaluation root seed: `9483999`. Evaluation used the common **original simulator reward**. The output says `formal_eligible=false`, `benchmark_eligible=false`, and `locked_test_access=false`; no reserved/locked outcome data were accessed.
- The cognition intervention results are paired first-decision probability probes, **not** paired full-trajectory causal effects. No biological-validity inference is supported.

## Real Kaggle runs

Both versions ran on Kaggle’s two NVIDIA Tesla T4 devices and observed Python `3.13.15`, NumPy `2.1.3`, PyTorch `2.11.0+cu128`, CUDA runtime `12.8`. Both notebooks printed `CUDA forward/backward: PASS`. Training metrics show CUDA execution and 32 updates/model. The training jobs and checkpoints completed on the GPU; only the notebook’s later evaluation/guard cells failed.

| Kaggle version | Training source commit | Models completed | Notebook terminal status | Recorded failure |
|---|---|---:|---|---|
| v12 | `d2fdecf1a49548056775ad0fecaff4aefc6542e3` | A–D, 12 | `ERROR` | At ~600s the evaluator rejected the report’s shared evaluation-root field (`underlying evaluation report did not preserve the shared evaluation root seed`). The jobs had reached generation 32 and exported their final checkpoints. |
| v13 | `58778e4469f112514326d0eaa616e4ed5e79b495` | E–F, 6 | `ERROR` | At ~266s an after-training notebook assertion expected an `formal_eligible` field that the evaluator does not emit (`Iteration-2 evaluation was not the expected disjoint exploratory evaluation`). The jobs had reached generation 32 and exported their final checkpoints. |

The post-training assertion was fixed and pushed in `06829290af0e236b8a3fc9eff426d41252231f7a`, and the comparison runner was updated to allow the separately fixed E/F seed triplet while continuing to enforce identical seeds within each treatment. The notebook was **not** re-executed after that fix; therefore v12/v13 remain failed notebook runs, not a clean notebook-level pass. The A–F comparison below was run locally against downloaded artifacts, not on reserved/formal data.

## Outcomes on common exploratory worlds

Values below aggregate each condition’s 36 trained episodes. Action frequencies are proportions of all animal decisions; resource successes are event counts over the same episodes. These are descriptive small-sample exploratory outcomes.

| Rank | Condition | Training treatment | Mean original return / episode | Graze | Drink | Herd | Rest | Forage successes | Water successes | Mean final hunger | Mean final thirst |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | B | Original reward; normalized advantage + entropy 0.01 | 4.130 | 0.0% | 0.6% | 47.5% | 24.5% | 0 | 4 | 0.997 | 1.000 |
| 2 | D | State-aware shaping scale 0.1; normalized advantage + entropy 0.01 | 1.637 | 0.1% | 1.1% | 50.0% | 9.5% | 0 | 6 | 0.990 | 1.000 |
| 3 | A | Original A2C and original reward | 0.373 | 0.1% | 1.2% | 35.2% | 27.1% | 0 | 5 | 0.990 | 1.000 |
| 4 | C | State-aware shaping scale 0.1; original A2C objective | -2.174 | 0.1% | 3.5% | 36.2% | 13.4% | 0 | 7 | 0.979 | 1.000 |
| 5 | E | Stronger state-aware shaping scale 0.5; original objective | -12.398 | 24.2% | 18.9% | 32.7% | 3.5% | 60 | 23 | 0.954 | 1.000 |
| 6 | F | Stronger shaping scale 0.5; normalized advantage + entropy 0.01 | -13.106 | 27.5% | 16.3% | 32.9% | 1.4% | 60 | 23 | 0.961 | 1.000 |

For completeness, the remaining A–D action shares (walk/ruminate) were A 11.6%/24.8%, B 9.4%/18.0%, C 12.7%/34.1%, D 11.1%/28.3%. E: 0.3%/20.3%; F: 0.9%/20.9%. Invalid actions were zero in all evaluated trained policies. Mean policy entropy was 1.682–1.704 nats; this did not prevent behavioral collapse.

E/F materially increased resource-directed actions and successes, but this did **not** meet the combined objective: mean returns were worse than their matched untrained controls (`-8.596` vs `-12.398/-13.106`), and final thirst remained fully saturated at 1.0. Hunger remained very high (0.954/0.961), and fatigue was 0.269/0.296. A–D had no forage-success events and nearly no grazing; herd interaction/rest remained dominant. The heuristic had high resource-seeking frequency (graze 57.8%, drink 22.7%) but lower mean reward (`-35.872`); this does not establish a superior policy.

**Interpretation:** the stronger state-aware treatment partially reduced the original action-collapse signature and improved resource-event counts, but achieved this by trading away common-reference return and still did not resolve thirst. It is not a usable improved policy under the stated success criterion. Do not select/promote E/F as winning policies. B ranked highest by return but did not graze and had only four water-success events.

## Training and hardware measurements

Per model the Kaggle run recorded 32 updates and 512 episodes. A–D took 74.2–102.9 seconds/model (mean 87.2s); E–F took 66.1–82.5s/model (mean 75.8s). Rollout simulation dominated: approximately 2.3–3.3 seconds/update; measured forward/loss/backward/optimizer time was generally 0.01–0.03 seconds/update, checkpoint writes ~0.016 seconds/update. Across conditions, raw rollout-batch return rose modestly from the first four to the last four updates for A–D, but the held-out common-world results still showed no forage successes. E/F raw batch return did not improve over the last four updates. Therefore training reward trends are not evidence of useful behavioral learning.

The single-repeat calibration batch had 16 episodes/5,760 decisions: CPU 3.458s (rollout 3.371s, forward/backward 0.087s), CUDA 0 3.640s, CUDA 1 3.620s, and two independent CUDA shards 3.849s/3.749s. This tiny, one-repeat measurement does not demonstrate end-to-end GPU speedup. Two independent training workers were scheduled on the two GPUs; no DDP was used. We make no GPU-acceleration claim.

## Cognition probes and limitations

The first-decision probes changed specified memory/perception inputs and measured policy probabilities. Across A–D, changing memory water-success/blockage features did not increase drink probability (effects approximately -0.002 to -0.004). Making water unavailable changed drink probability by only about -0.034 to -0.037. The prescribed descriptive +0.05 margin was not met. E/F showed larger but inconsistent effects, not a replicated trajectory-level causal result. Ablating the corresponding feature groups resulted in little/no attenuation in measured drink probability for several probes. Treat as a warning of weakly grounded cognition, not proof that cognition has no effect.

The A–F evaluator output records the scenario/world hashes, outputs, reward components, action frequencies, state outcomes, invalid-action counts, paired probe inputs and probe hashes. It is a small exploratory sample and does not include formal clustered/Holm inference, reserved outcomes, or biological validation.

## Downloaded model/artifact integrity and local usability

All **18 final resumable PyTorch generations** and all **18 NumPy-compatible inference exports** were downloaded. Each generation was restored locally with the project checkpoint loader; its generation/state SHA-256 matched the per-run artifact manifest; each state had episode cursor 512. Each `.npz` and JSON sidecar hash also matched its Kaggle artifact manifest and loaded through `farm_sim.policy.load_checkpoint` in the local CPU evaluation environment. The bundle contains 144 separately SHA-256 checked files (including logs, worker-shard summaries, calibrations, full final checkpoints, inference exports, metrics, and per-model evaluation reports).

- Bundle: [`model-artifacts.zip`](model-artifacts.zip) (683,840 bytes; SHA-256 `03cdb5a0aed8ce5e670d1bf3e2d60c5a0db0eeb8691d7d258da392a346f7951e`)
- Bundle file manifest: [`artifact-manifest.json`](artifact-manifest.json)
- Machine-readable comparison: [`exploratory-comparison.json`](exploratory-comparison.json) (SHA-256 `df62812358ff5b6ac8620689a3e769af1c093c28c59f6b449756ce9db569866a`)
- Plot: [`exploratory-comparison.svg`](exploratory-comparison.svg)
- Raw Kaggle output logs are retained inside the bundle.

Extract the bundle and load an inference export as follows:

```python
from farm_sim.policy import load_checkpoint

policy = load_checkpoint("A/seed-9483001/torch-a2c-seed-9483001.npz")
decision = policy.decide(feature_vector)  # feature_vector uses farm_sim FEATURE_NAMES order
```

The simulator’s normal policy-loading path accepts these `.npz` exports; retain the JSON sidecar beside each export. The `.pt` generation files are resumable training states, not the simulator’s inference format.

## Reproduction and verification

The exact Kaggle training sources remain on branch `exploratory-a2c`; source commits and captured Kaggle logs are in the bundle. Comparison was regenerated using the committed checkpoint manifest and common evaluation seed `9483999`, three episodes per scenario family:

```bash
PYTHONPATH=. python -m validation.compare_exploratory_checkpoints \
  --checkpoints /tmp/pasture-exploratory-all-18-checkpoints.json \
  --eval-seed 9483999 --episodes-per-family 3 --device cpu \
  --output validation/kaggle-exploratory/reproduced
python3 scripts/kaggle_deploy.py verify \
  validation/kaggle-exploratory/artifact-manifest.json \
  --artifacts /path/to/extracted/pasture-exploratory-model-artifacts
```

Local evaluation used Python 3.12.3, NumPy 1.26.4, PyTorch 2.11.0 CPU. This was evaluation and checkpoint-load verification only; no model training was run locally. `tests/test_exploratory_learning.py`, `tests/test_exploratory_checkpoint_comparison.py`, `tests/test_exploratory_batch_cli.py`, `tests/test_kaggle_deploy.py`, `tests/test_kaggle_release.py`, `tests/test_torch_training.py`, and `tests/test_torch_evaluation.py` were run. At the time of this report, 64 passed, 25 skipped, and one deployment test failed because the working tree was intentionally dirty (it enforces a clean committed source checkout before staging); the same suite must be rerun after committing this report and comparator update.

## Verdict

**Training execution: PASS per 18 individual Kaggle training jobs; NOTE both notebook versions ended ERROR after training.**

**Artifact download/hash/load: PASS (18/18 PyTorch checkpoints and 18/18 inference exports).**

**Exploratory comparison and held-out common-world evaluation: PASS as descriptive exploratory analysis.**

**Requested behavioral-improvement gate: FAIL.** E/F increased grazing/drinking and forage/water event counts but had worse original reward and persistent maximal thirst. No candidate demonstrated both improved resource behavior and nonworse overall state/reward. Preserve the evidence and do not promote any of these policies as a validated improvement.

**Formal cognition, locked-test, and biological validation: NOT RUN / NOT CLAIMED.**
