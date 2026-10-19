# Experimental Release Engineering Follow-up

**Current verdict: engineering implementation progress verified; full experimental release NOT READY.** This follow-up adds only outcome-free implementation, synthetic tests, and provenance review. It performed no precision-pilot episodes, full-scale training, held-out scientific evaluations, or locked-test access. It does not alter the frozen rewards, seeds, protocol thresholds, historical failed evidence, or biological-validity status.

## Baseline evidence retained

The immutable prior bounded READINESS result remains [KAGGLE_EXPERIMENTAL_RELEASE_READINESS_REPORT.md](KAGGLE_EXPERIMENTAL_RELEASE_READINESS_REPORT.md): Kaggle private notebook v8 completed on two Tesla T4 GPUs; real CUDA forward/backward, bounded DDP work on each GPU, same-kernel fresh-process checkpoint resume, and 7/7 downloaded artifact checks passed. The source release was `fc0b9e594ff7e2caa0aca94e6321b5a9b98ac2b9`, runtime Python 3.13.15 / NumPy 2.1.3 / PyTorch 2.11.0+cu128 / CUDA 12.8. Those are bounded engineering claims, not scientific training, GPU speedup, or session-loss recovery evidence.

## New local draft artifacts

- `validation/run_torch_precision_pilot.py` verifies source-pin inclusion/hash equality before generating an outcome-free schedule. It has no production execution or artifact-writing helper; the only statistics helper consumes fabricated numeric arrays and returns a value explicitly marked `synthetic_test_only`. The production Python API and CLI execution mode fail closed because this protocol lacks independent approval and a trusted signed-authorization workflow. The current v1 preregistration does not bind the newly added runner, so it is not executable with that historical manifest.
- `validation/finalize_precision_pilot_v2.py` is a distinct finalization boundary. It rejects five-field caller-supplied SD artifacts, requires a signed independent protocol-review approval, signed pilot authorization, and independent signed run attestation from three distinct signer keys. It fingerprints canonical DER public-key identity (not PEM bytes), verifies the exact allowlisted key snapshot without reopening a replaceable path, validates full source/runtime/notebook/schedule bindings, records access before reading raw cells, hashes exact bytes before parsing, recomputes sample SD and the existing `delta0 = 0.05 * SD`, and emits only masked statistics. The audit store is fixed in code rather than caller-selected to prevent trivial replay; it still requires independently protected durable append-only storage because a hash chain cannot prevent whole-log replacement. All three trust-root allowlists are empty and the candidate remains unapproved, so production calls deny before reading pilot results. Synthetic Ed25519 fixtures exercise only fabricated data.
- `validation/torch_locked_statistics.py` implements crossed checkpoint × world resampling, one-sided centered bootstrap tests, Holm adjustment, and Holm-inverted lower bounds. If the attainable p-value resolution cannot reject under Holm, the lower bound is absent (`None`) rather than an exact-looking point estimate.
- `validation/analyze_torch_cohort.py` implements the approved nine utility/algorithm tests and separate four cognition tests; it validates complete paired 10-checkpoint × registered-world cohorts and explicit source/checkpoint/audit/selection bindings. It also retains descriptive action distributions/collapse flags, invalid counts, reward trajectories, final physiology, resources, and navigation reasons. Its file loader authenticates first, checks that the freeze binds the exact analysis sources, validation selection and bootstrap configuration, appends/fsyncs an audit record before reading evaluator payloads, verifies each per-checkpoint evaluation/access binding, and compares exact evaluator-document bytes against a post-evaluation hash-chained artifact record before parsing. It rejects the current v1 freeze because it omits these new source hashes and selection binding; the pinned trusted authorization root remains empty. The hash chain still needs operational tamper-resistant storage; hash chaining alone cannot prevent whole-log replacement.
- `validation/protocols/pasture-study-analysis-v2-candidate.json` records exact source hashes and the immutable Kaggle v8 publication/runtime/protocol baseline. It is explicitly dirty-worktree, DRAFT, non-releaseable, and has all authorization flags false. Its canonical SHA is inside the candidate. It does not replace or edit v1 evidence.

## Verification evidence

Synthetic-only targeted tests:

```text
.venv/bin/python -m unittest \
  tests.test_torch_precision_pilot \
  tests.test_torch_locked_statistics \
  tests.test_torch_cohort_analysis \
  tests.test_torch_release_candidate -v
```

The focused implementation tests cover frozen source pins, synthetic-only sample-SD edge cases, denied pilot execution by CLI and Python API, signed synthetic v2 finalization, exact raw-byte attestation, empty trusted-root denial, duplicate/incomplete/non-finite/extra cells, SD-only artifact rejection, reproducibility and Holm resolution, exact nine-/four-test family sizes, paired cohort completeness, strict seed types, intervention target-feature fidelity, bit-identical sham, post-evaluation artifact digest binding, and fail-closed behavior before payload parsing when authorization is missing. These fixtures are fabricated test data only.

Historical full local verification before the v2 finalizer hardening:

```text
.venv/bin/python -m pytest -q
192 passed, 2 skipped, 162 subtests passed in 80.44s

.venv/bin/python -m unittest tests.test_torch_release_candidate \
  tests.test_torch_cohort_analysis tests.test_torch_precision_pilot \
  tests.test_torch_locked_statistics tests.test_torch_evaluation -q
33 tests passed (before the separate v2-finalizer implementation)

.venv/bin/python -m py_compile validation/analyze_torch_cohort.py \
  validation/evaluate_torch_mvp1.py validation/run_torch_precision_pilot.py \
  validation/torch_locked_statistics.py
git diff --check
PASS
```

During iteration, one full run caught a stale candidate hash after a source edit; the candidate was regenerated, its integrity test passed, and the final full regression above passed cleanly. The CLI denial message emitted by the synthetic pilot denial test is expected. No Kaggle notebook version was published or executed for this follow-up.

Latest clean full regression after the key-identity and replay/tamper test updates:

```text
.venv/bin/python -m unittest tests.test_torch_precision_pilot_v2 -v
6 tests passed

.venv/bin/python -m pytest -q
198 passed, 2 skipped, 162 subtests passed in 80.42s
```

The latest synthetic fixtures verify distinct DER key identities (including PEM re-encoding), replay denial through the fixed store, and raw-byte tamper detection. Candidate hashes were refreshed after report/source edits and their integrity test passed. The external audit-store protection and all real trust provisioning remain unverified and deliberately disabled.

## Readiness gates

| Gate | Current result | Evidence / remaining issue |
|---|---|---|
| Baseline protocol, historical failure reports, rewards and thresholds | **PASS / preserved** | v1 preregistration, threshold hash, historical MVP1 reports, requirements pins, and Kaggle v8 bundle were not edited. |
| Kaggle T4x2 bounded execution / CUDA / checkpoint restore within a kernel | **PASS (prior bounded readiness only)** | See immutable v8 report. Recovery after Kaggle session loss remains unverified. |
| Crossed bootstrap, Holm tests, exact family sizes | **PASS on synthetic tests** | No study outcomes were read. This does not validate inferential performance on actual outcomes. |
| Cohort completeness, pairing, causal probe fidelity, descriptive diagnostics | **PASS on synthetic tests** | Analyzer requires explicit caller freeze bindings; locked-file loader still rejects the current v1 freeze and cannot pass while trust root is empty. |
| Precision-pilot runner source binding and output masking | **PARTIAL / BLOCKED** | Production runner remains intentionally disabled. Historical v1 finalizer still accepts unverified five-field SD artifacts and must not be used for real outcomes. A v2 sidecar/attestation finalizer is now implemented and covered by synthetic signed success/tamper tests, but independent review, trusted roots, and protocol authorization are still absent. |
| Authenticated cohort loader / pre-read audit | **FAIL CLOSED / NOT READY** | Loader implementation is present but cannot authorize: the trusted signer allowlist is intentionally empty, v1 freeze lacks new analysis hashes and validation-selected cognition arm, and current authorization contract is not yet a reviewed v2 contract. |
| Immutable experimental release / Kaggle publication | **NOT READY** | New source is uncommitted and unpublished. The candidate binds hashes but explicitly cannot claim a clean source commit. The old notebook hard-disables EXPERIMENT/FULL_VALIDATION. |
| Full REINFORCE vs A2C training / validation learning curves | **NOT RUN** | No scientific outcomes authorized or generated. WALK/HERD_INTERACT collapse evidence remains untouched. |
| Locked utility/cognition tests and δ0 freeze | **NOT RUN** | No precision pilot, δ0, selected candidate, or locked outcomes. |
| GPU acceleration / biological validity | **NOT CLAIMED / NOT VALIDATED** | Presence/participation is not speedup; no biological validation. |

## Independent review

- Code/security review of the latest audit/pilot changes: **PASS for reviewed code scope**. The reviewer verified the DER public-key fingerprint fix, the alternate-PEM same-key rejection test, exact key snapshot use, and found no remaining high/medium code finding. Before enabling any roots, the fixed audit path must be provisioned in independently protected durable storage; this was not verified and is not a release approval.
- Adversarial QA: **PASS for removal of the identified pilot-helper bypass; overall readiness FAIL/BLOCKED**. It independently confirmed the v1 finalizer still trusts caller-supplied five-field SD artifacts, and that the hash-chained audit needs external protection against wholesale rewrite.
- Architecture review: **BLOCK** for experimental release. Required missing items are a reviewed v2 protocol and sidecar-verifying finalizer, governance-approved trusted signer/root plus tamper-evident storage, clean source commit/new immutable publication, and actual Kaggle session-loss recovery evidence.
- v2 pilot-finalizer architecture review (read-only design review before implementation): recommended exact signed authorization/attestation and cell-schema boundary; implementation now follows that design in code but still needs a fresh adversarial/code review.
- Follow-up hardening: TOCTOU, caller-selected replay, signer-role reuse, and PEM-encoding identity findings were addressed and independently re-reviewed. The fresh full regression passed; no real outcome data was accessed.

## Required next gates

1. Independent statistical/security review and explicit approval of a separately versioned protocol that binds a production pilot runner, its output/sidecar verifier, analysis implementation, access authorization, exact source/runtime configuration, and bootstrap config. Keep historical v1 byte-for-byte evidence unchanged.
2. Obtain fresh independent review of `validation/finalize_precision_pilot_v2.py`, then keep the historical v1 finalizer restricted to fixtures only. The v2 finalizer rejects caller-supplied SDs and verifies review/authorization/attestation via distinct keys, but it is not executable under the current empty signer allowlists or unapproved candidate. A production pilot runner remains to be separately implemented/reviewed if governance approves pilot collection.
3. Governance must configure a trusted signer through an independently approved code/policy change and a durable append-only audit store. Neither approval nor trust is inferred from this task or this draft candidate.
4. Only after those gates and separate authorization: run the masked precision pilot, independently verify the artifact/sidecar/source/runtime hashes, freeze δ0 and validation-only checkpoint/arm selection, then perform separately authorized study training and only later locked evaluation.
5. Publish a new private Kaggle notebook revision only after a clean source commit and new release manifest agree. Run only PREFLIGHT/SMOKE/READINESS while those are the authorized modes. The current notebook provides no valid EXPERIMENT/FULL_VALIDATION command.

## Reproduction commands and stop boundary

Run only local synthetic regression now:

```bash
.venv/bin/python -m unittest \
  tests.test_torch_precision_pilot tests.test_torch_locked_statistics \
  tests.test_torch_cohort_analysis tests.test_torch_release_candidate -v
.venv/bin/python -m pytest -q
```

The existing exact Kaggle CLI publish/status/log/output/hash commands for bounded READINESS are in `docs/kaggle-training.md` and the v8 report. Do not use them to select `EXPERIMENT` or `FULL_VALIDATION`; the notebook rejects those values. No full-training, pilot-execution, or locked-analysis command is authorized or made available by this draft. **Stop before collecting any scientific outcomes.**
