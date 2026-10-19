# Pasture Studies — exploratory A2C Kaggle run v11

**Result: execution PASS; exploratory behavior is weak and is not a learning/cognition acceptance result.**

- Kaggle notebook: <https://www.kaggle.com/code/pascalcoimbra/pasture-studies/versions/11>
- Kaggle status: `KernelWorkerStatus.COMPLETE`
- Kaggle CLI: `2.2.4`
- Source commit pinned by notebook: `73e4ae05af06e14cc27901a54d0c2b652c1a73b7`
- Release manifest self-hash (generator): `eac853ccf67895b07e9c773d4717a67ce010c21c278dff77b32364e7ee418a64`
- Bound manifest-file SHA-256 (deployment/attestation): `ddf95dd3e64b12b5a34695e86a0df4e9579bafba130e6153ce02ad6e6692df29`
- Content attestation: `validation/kaggle-exploratory/kaggle-v11-execution-attestation.json` (`exact_match: true`)
- Runtime: Python `3.13.15`, NumPy `2.1.3`, PyTorch `2.11.0+cu128`, CUDA `12.8`; 2 visible NVIDIA Tesla T4 GPUs. The bounded learner used **one GPU (`cuda:0`)**; the second GPU was detected, not used for this single-process run.

## Bounded workload actually run

- Mode: `EXPLORATORY` only; A2C, seed `9482817`, 256 complete episodes, 16 optimizer updates, 5 generated descriptive rollout worlds.
- No `EXPERIMENT`, `FULL_VALIDATION`, preregistered split, held-out comparison, locked-test analysis, selection, or promotion was run.
- CUDA forward/backward smoke: PASS. The learner reported CUDA available and produced all 16 optimizer updates.
- Learner wall time: `50.999 s` (notebook log shows approximately `74.2 s` through artifact creation).
- Five evaluation worlds × 72 ticks × 5 cattle = 1,800 recorded evaluation decisions. Each world completed; every animal has per-animal action, reward, final state, and memory-size fields.
- All 16 updates report zero invalid actions. The sampled training action counts total 92,160 decisions: drink 25,237; graze 11,849; herd-interact 13,116; rest 18,685; ruminate 10,774; walk 12,499.

Update reward trajectory (descriptive batch returns; not a formal metric):

`-10.989, -10.572, -9.717, -10.186, -9.852, -10.371, -9.244, -9.128, -9.213, -9.352, -9.234, -8.588, -8.643, -8.398, -9.013, -8.460`

Mean of first four updates: `-10.366`; last four: `-8.629`. The curve is noisy and mostly negative; this small-run trend does **not** establish learning efficacy.

## Behavior inspection and limitations

Deterministic evaluation over the five generated worlds yielded returns `4.462, 2.671, -21.119, 5.413, 4.608` (mean `-0.793`). Aggregate actions: rest 1,410/1,800 (78.3%), drink 319, graze 58, walk 13, herd-interact 0, ruminate 0. This is a conspicuously skewed evaluation policy with no herd interaction or rumination in these rollouts; it should not be presented as useful cattle behavior. Reward/state values are synthetic software-task outputs, not biological measurements.

The exported NumPy policy was loaded by the local farm server and exercised through its HTTP simulation API for 72 ticks with all five cattle (360 decisions): rest 140, drink 177, graze 36, walk 7; no herd-interact or ruminate. The endpoint returned five valid live animal records at every tick. Detailed API evidence is in `kaggle-v11-local-simulation-smoke.json`. Hunger/thirst fields ended at `1.0` for all five; without biological calibration these are only simulator state values and indicate a need for follow-up debugging, not animal welfare evidence.

## Checkpoint and evidence integrity

- Latest checkpoint: `artifacts/exploratory/kaggle-v11/exploratory-only/exploratory-only/a2c/generation-00000016/`
- Inference export (loads in local simulator): `artifacts/exploratory/kaggle-v11/exploratory-only/exploratory-only/a2c/torch-a2c-seed-9482817.npz`
- Policy sidecar: same basename with `.json`; identifier `torch-9482817-256`.
- There are 16 integrity-checked checkpoint generations, `LATEST` points at generation 16, and a 38-file SHA-256 artifact manifest.
- `scripts/kaggle_deploy.py verify .../artifact-manifest.json --artifacts ...` passed all 38 entries. ZIP test passed with 39 entries (manifest + 38 files).
- `pasture-artifacts.zip` SHA-256: `c3b28c0bd598272bd305cb143dcfd472345f14594142021e0e1075fb4998a658`
- `.npz` policy SHA-256: `091d45068ba30d6040737107c8e0d597948cbae9f7a1e276945853e5da428789`
- Full Kaggle log: `validation/kaggle-exploratory/kaggle-v11-run.log`.

## Reserved-data isolation and non-final attempts

The v11 notebook used a blob-filtered, sparse checkout containing only `farm_sim/`, `validation/train_torch_mvp1.py`, `scripts/kaggle_release.py`, `requirements-training.txt`, and the notebook itself. A runtime guard fails if `artifacts/` or `validation/kaggle-mvp1/` is present. A separate local clone of the same immutable commit verified the sparse checkout materializes the required code but excludes those directories. The training code did not load any reserved split.

For transparency: v9 trained all 16 updates but ended `ERROR` in a post-training check because the notebook looked for `metrics.jsonl` in the wrong directory. v10 completed but its broad checkout materialized the repository's archived artifacts, so it was superseded for isolation reasons. Neither is treated as the final result. Logs are retained as `kaggle-v9-failed-run.log` and `kaggle-v10-superseded-run.log`; v11 is the final isolated run.

This is an exploratory software run only: no baseline comparison, causal intervention, biological validation, learning acceptance gate, or cognition claim was performed. Existing formal criteria and historical evidence remain unchanged.

## Local regression verification

After the run implementation and evidence were committed, the full local suite passed: `174 passed, 2 skipped, 146 subtests passed` (`pytest -q`, 82.29 s). The local CUDA environment is CPU-only; this result does not include a local GPU benchmark. The final work is on public branch `exploratory-a2c`; latest report/evidence commit: `cb026b3a0755b36671ec7d4bf2de4bc660bfaf63`.
