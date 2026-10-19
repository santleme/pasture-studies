# Pasture Studies integration audit — 2026-10-19

## Delivered branch

- Branch: `integration/pasture-studies-20261019`
- Worktree: `/home/lucas_coimbra/projects/experiments/riose-tech-integration`
- Integration merge tip: `d485b64`; the audit note is committed afterward (see
  `git log` for the final branch HEAD). No push and no merge to `main`.
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

## Follow-up integration audit — 2026-10-19

This section supersedes the earlier “not integrated” status for compatible
uncommitted main-worktree changes. It records the follow-up integration requested
by the user. The source main worktree and stash are still unchanged.

### Session and OMX ownership check

Before transporting files, I inspected all three tmux OMX sessions and the main
worktree. The current integration session was attached; the older dataset Team
session had a complete manifest (4 complete, 0 pending/in-progress); the other
detached Codex session (`omx-1792426377062-vjxj7w`, pane `%39`) was alive but
showed an idle prompt with no active command. I did not send input to or stop
any session. The earlier read-only integration Team still has unresolved
pending tasks and lost worker-pane identity; its detached worker worktrees remain
untouched at the main base. Its artifacts were not treated as completed source
work. A failed `omx ultragoal create-goals` reported that the current
`session.json` was unusable and no writable lifecycle scope could be resolved;
no OMX goal/state files were edited as a workaround.

### Changes integrated

Two follow-up commits were added after the earlier audit:

- `490e5ae Integrate DAgger generalization and RL pilot evidence`
- `b14f030 Add fail-closed precision analysis and audit binding`

They add the main-worktree-only DAgger causal/generalization metrics and test,
solvability-v2 and pilot reports, optional PettingZoo/SB3 requirements, PPO
Gymnasium API compatibility, locked-evaluation post-write audit binding, and the
synthetic masked precision-pilot/statistical/cohort-analysis candidate with
fail-closed v2 finalization. The candidate source hashes were rebound to the
integrated implementation; its independent review and all pilot/training/
locked-test authorization flags remain false, and `release_eligible` remains
false. No historical preregistration, reward, threshold, or locked result was
rewritten.

Existing integration versions were retained rather than replaced where they
were already equal or more complete. In particular, dataset DAgger already had
two aggregation rounds plus a Kaggle workload guard and `deepcopy`-reset worlds;
this is safer than the dirty-main copy, which omitted those safeguards. The
existing guarded autoresearch/SB3/DAgger CLI versions were also kept; dirty-main
changes that removed Kaggle guards were excluded. The latest DAgger report now
includes action distributions, and its resource-preview test was added without
discarding the prior tests.

The Kaggle execution-boundary documentation and the new experimental-analysis
candidate section were combined, not substituted. The PPO CLI changes preserve
the existing Kaggle gate, adapt to `PastureEnv(controlled_animals=...)`, and
lazily import the adapter for random-policy evaluation. Dataset packaging,
benchmark seed/split checks, previous Kaggle manifests, and the exploratory A2C
work remain intact.

### Exclusions and preservation

- `.codex/` and its associated `.gitignore` relaxation were not imported. The
  directory contains live session transcripts, SQLite state/queue databases,
  WAL/SHM files, plus local OMX profiles/prompts/skills; none is Pasture source,
  and copying it would conflate local session/runtime data with project code.
- Dirty-main edits that weaken/remove local Kaggle workload guards in
  `validation/autoresearch_pasture_rl.py`, `validation/run_dagger_generalization.py`,
  `validation/sb3_pilot.py`, and `validation/sb3_single_agent_followup.py` were
  excluded. The integration branch retains the stronger guarded implementations.
- Files already identical on the integration branch were not copied again. The
  dataset-generation and policy-summary tests were already integrated; local
  DAgger dataset generation tests were not re-run because they perform local
  supervised fitting.
- `stash@{0}` was not applied, dropped, rewritten, or otherwise changed. Its
  `.gitignore`, Kaggle docs, evaluator, and evaluator-test hunks overlap the
  dirty main snapshot. The compatible docs/evaluator/test changes were brought
  from the main working tree through reviewed patches; the local OMX ignore
  relaxation was excluded. The stash remains available verbatim.
- All main-worktree edits remain present in place. `main` still points to
  `7ed39ca` and remains dirty; the follow-up imported copies only into this
  branch's isolated worktree. No worktree, branch, stash, or remote ref was
  removed, reset, pushed, or merged into `main`.

### Verification and deviation

- Static compilation passed:
  `/home/lucas_coimbra/projects/experiments/riose-tech/.venv/bin/python -m compileall -q farm_sim scripts validation tests`.
- The final focused, non-training verification passed **54 tests** across
  Gymnasium contract/random-policy evaluation, solvability-v2, synthetic pilot
  and locked statistics/cohort analysis, artifact summaries, evaluator audit
  binding, and execution-policy gates. The source-hash candidate test initially
  failed after the docs merge; after rebinding candidate hashes it passed.
- **Process deviation:** an earlier broad test selection unintentionally ran
  the pre-existing SB3 optimizer-smoke test before I noticed it. It performed
  eight PPO timesteps for each of one- and five-animal environments on CPU,
  with in-memory models only and no checkpoint/model output retained. I stopped
  that broad selection; the final 54-test run explicitly selected tests that do
  not optimize/train, and no dataset-generation or research-training tests were
  run. This accidental smoke run did not satisfy the user's no-training
  constraint and is disclosed rather than represented as compliant.
- `git diff --check` passed before commit. No test submitted Kaggle jobs, accessed
  locked outcomes, published datasets, or altered preregistration evidence.

The integration branch tip after this audit is recorded by `git log`; it remains
local-only. No push, main merge, worktree cleanup, stash operation, or training
run was authorized or performed apart from the disclosed test-smoke deviation.

### Independent review follow-up

The post-integration independent review identified two medium verification issues
and one Kaggle boundary gap. They were corrected before delivery:

- Direct calls to `train_generalized_dagger()` and `train_ppo_finetune()` now
  enforce the Kaggle location policy. The only local DAgger escape is an
  explicit one-initial-episode / one-round / one-episode smoke envelope; the
  study CLI passes its already-validated smoke flag through. A direct-call test
  proves both full-training APIs deny before data collection or PPO import.
- The precision-pilot manifest loader now reads bytes once, validates the hash
  and parses those same bytes; it no longer reopens the file after validation.
- The autoresearch and generalization gates now require caller-supplied evidence
  paths instead of depending on ignored `.omx/specs/` files. Their CLI `--help`
  checks show the inputs as explicit required parameters; no private OMX
  evidence was copied into source.
- The draft candidate hashes were rebound after the precision-loader change;
  current candidate hash is `cdac0c33439a293c72e31142b79f3fe7f402ca6af8203823af6e31766a9b8196`.
  Its authorization booleans remain false.

After these fixes, `compileall` passed, the focused non-training unit suite
passed **55 tests**, and the three targeted review-regression pytest cases passed.
The broad smoke test that was accidentally invoked earlier remains the only
reported deviation; no later check ran an optimizer or training/data-generation
loop. Review findings will be marked resolved only after both independent lanes
review this corrected tip.

Review verdict for the corrected integration: the code-reviewer returned
**APPROVE** with zero remaining findings; the architect returned **WATCH** only
because the separate precision/locked-analysis candidate is deliberately not
release-ready (no trusted signer roots and no independently protected audit
store). Under the review contract the synthesis is **COMMENT**, not unconditional
merge approval. This does not block the requested local integration branch, but
it does block any claim that the v2 pilot/locked-analysis path is ready for
production or authorized execution. Both lanes reviewed the corrected code tip
`b87370d`; the final commit after it changes only this audit note.
