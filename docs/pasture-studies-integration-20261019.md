# Pasture Studies integration audit — 2026-10-19

## Delivered branch

- Branch: `integration/pasture-studies-20261019`
- Worktree: `/home/lucas_coimbra/projects/experiments/riose-tech-integration`
- HEAD: `d485b64` (integration branch; no push and no merge to `main`)
- The original `main` worktree, its uncommitted files, the source branches, and
  stash remain untouched.

## Integrated work and order

1. Fast-forwarded from `7ed39ca` to
   `codex/pasture-kaggle-execution-boundary` (`0d8e7ce`), adding the fail-closed
   Kaggle execution policy and DAgger/PPO/PettingZoo pilot and validation code.
2. Merged `codex/pasture-datasets-benchmarks` (`3b550a7`) as `3a53c37`:
   deterministic two-round DAgger dataset aggregation, exact frozen validation
   seed-map checks, disjoint-split enforcement, and corresponding tests/docs.
3. Merged `exploratory-a2c` (`1247b71`) as `d485b64`, preserving its
   exploratory A2C code, protocols, tests, Kaggle manifests, reports, and
   recorded model/evaluation artifacts. The missing `/tmp` worktree did not
   prevent integrating the committed branch history.
4. `feat/kaggle-benchmarks` (`3200f58`) is already an ancestor of local `main`
   and therefore included in the integration base.

The only source conflict was in `validation/train_torch_mvp1.py`. Resolution
kept the exploratory training parameters and routes while retaining the
Kaggle-only workload guard from the execution-boundary branch. Exploratory
checkpoints remain explicitly non-formal and non-benchmark eligible.

## Compatibility and verification

- The integration does not rewrite historical `make_world()` or the core
  transition semantics. DAgger/PPO/A2C experiments remain software-task
  experiments; no biological or optimal-expert claims were added.
- Benchmark summarization now requires the preregistered four-family validation
  seed map; dataset packaging remains training-only. The Kaggle boundary still
  guards extensive local training/evaluation entry points.
- Static check: `python -m compileall -q farm_sim scripts validation` passed.
- Focused integration/security/reporting suite: **73 passed, 29 subtests passed**.
- The first focused test attempt had one expected staging failure because the
  release gate requires a clean committed checkout. It passed after committing
  the integration merge. No training command was run. Dataset-generation and
  optimizer/smoke-training tests were intentionally omitted because they run
  local supervised/RL updates.

## Preserved blockers / not integrated

- The original `main` worktree has active uncommitted changes (12 modified
  tracked files plus untracked `.codex/`, RL/DAgger tests, runners and reports).
  They were not copied or committed because their completion/ownership cannot
  be established from Git evidence. They must be reviewed and committed or
  explicitly excluded before claiming a complete integration of all work.
- `stash@{0}` is preserved. It has four-file changes to `.gitignore`, Kaggle
  training docs, torch evaluation tests and the torch evaluator, overlapping the
  dirty main worktree; applying it could replace newer work.
- The old exploratory worktree and three detached worker worktrees under
  `/tmp/pasture-exploratory` are missing and marked prunable. Their recorded
  commits are retained where reachable from `exploratory-a2c`; unknown
  uncommitted bytes from the missing directories are not recoverable from the
  worktree metadata alone. No worktree cleanup/prune was run.
- The requested Team runtime failed before worker ACK because pane identity
  proof was lost. Its exact Team state/worktrees remain untouched with tasks
  pending; they were neither restarted nor cleaned. The integration and test
  review was completed by the leader, not by workers.

No branches, stashes, worktrees, or remote refs were deleted, reset, force-pushed,
or merged into `main`.
