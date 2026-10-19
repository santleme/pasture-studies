# Pasture Studies Kaggle Experimental Release Readiness Report

**Verdict: PASS for bounded engineering readiness; FAIL/BLOCKED for the full scientific experimental release.** The Kaggle T4x2 bounded READINESS run completed and all exported diagnostic artifacts verified. This does not authorize or demonstrate full REINFORCE/A2C experiments, utility/cognition acceptance, GPU acceleration, or biological validity. No full-scale training, precision pilot, held-out scientific evaluation, locked-test access, threshold change, or MVP2 work was performed.

## Release identity and provenance

- Kaggle notebook: <https://www.kaggle.com/code/pascalcoimbra/pasture-studies>, **version 8**, status `COMPLETE`, private.
- READINESS source commit: `fc0b9e594ff7e2caa0aca94e6321b5a9b98ac2b9` (`Harden Kaggle version verification`); the notebook fetched and checked out that exact commit.
- Release manifest file SHA-256: `4c85eb477139951b9d4b77b9498cf2d3023f00865da752e18323b68b13e013db`. Its internal canonical release-manifest digest is `d65ac05586d82e3f8d50ee4dcd1ddb465cebd683a12de98e3a75db5e03fe12bd`. Status remains `READY_PENDING_INDEPENDENT_APPROVAL`; it is evidence, not training authorization.
- Committed source notebook SHA-256: `d216d933bfde053e30c52797d3463863928c70d8fa4618ce10d2bf11efe34993`.
- Staged notebook raw SHA-256: `a6e7df9797a786fe71e350f14b683722c4d62423ee9117d206eda6009c233c42`; Kaggle-pulled raw SHA-256: `ac89a049d13d099b87067be30272dc7b2c3e673f2afd262fdb9988c76c83671d`. Raw bytes differ because Kaggle serialized notebook cell source as strings instead of line arrays. The full parsed notebook canonical-content digest and cell-source digest matched exactly; the host-side publication attestation records both raw hashes and the canonical match.
- Detached publication attestation: `validation/kaggle-readiness-v8/publication-attestation.json`, binding Kaggle version 8, source commit, release-manifest file hash, staged and remote raw hashes, canonical notebook content hash, and cell source hash.
- Kaggle metadata response for version 8: private `true`, GPU `true`, Internet `true`, machine shape `NvidiaTeslaT4` (T4x2), Python notebook.
- Official Kaggle CLI used: reported version `2.2.4`, isolated source commit `6894864ff4095cf98e9e36d3a694ff5fc9ba3818`, `--no-run` advertised. This source includes the merged version-aware pull fix; the official upstream PR reports live API checks for version-specific operations [Kaggle CLI PR #1201](https://github.com/Kaggle/kaggle-cli/pull/1201). The globally installed 2.2.4 binary was not used for publication because it did not advertise `--no-run` or implement the version-specific pull behavior.

## Real Kaggle READINESS results

Observed runtime: **Python 3.13.15; NumPy 2.1.3; PyTorch 2.11.0+cu128; CUDA runtime 12.8; two Tesla T4 GPUs.** Historical `requirements-training.txt` remains unchanged (`numpy==1.26.4`, `torch==2.14.0`); no library reinstall was performed. The Kaggle runtime was separately recorded rather than treated as the historical scientific environment.

- CUDA availability and a real forward/backward operation: **PASS**.
- T4x2 distributed path: **PASS for bounded READINESS**. Two NCCL ranks mapped to `cuda:0` / `cuda:1`. Each rank processed 8 episodes per update, reported finite nonzero gradients and an optimizer step that changed model parameters; post-step model digests agreed across ranks for each update.
- Interrupted-process recovery: **PASS within the same Kaggle kernel session**. A first `torchrun` process committed update 1 / checkpoint generation 1. A fresh `torchrun` invocation resumed `LATEST`, committed update 2 / generation 2, and advanced the episode cursor from 16 to 32. This does **not** establish recovery after Kaggle terminates the kernel or deletes `/kaggle/working`.
- Workload: A2C readiness-only, diagnostic seed `37331`, 32 diagnostic episodes, two updates, no evaluation. Diagnostic reward sums were `-153.9755` and `-142.9108`; they are explicitly labeled non-evaluation diagnostics and are not a learning result.
- Notebook diagnostic elapsed time: **26.734 seconds** for the bounded READINESS training section. This is not a benchmark or a GPU speedup measurement.
- Artifact verification: **7/7 manifest-listed files verified**, including `LATEST`, two generation manifests, two training checkpoints, readiness evidence, and readiness report. ZIP tested and downloaded SHA-256: `0892f16ee045d45c8cf07e2e78e4f297a903acaef932a30e0b9294e821c2332f`.
- All readiness/checkpoint records set `benchmark_eligible=false`, `evaluation_performed=false`, and `locked_split_accessed=false`.
- Logs contain non-fatal debugger frozen-module messages, PyTorch DDP barrier device-context warnings, OMP thread notices, and Python/nbconvert syntax warnings. The run reached `COMPLETE` and the integrity checks passed.

The preserved evidence bundle is `validation/kaggle-readiness-v8/`. Its `artifact-manifest.json`, `readiness-evidence.json`, checkpoint generations, zip, Kaggle logs/status, runtime metadata, release manifest, and detached deployment/publication records are included. `kaggle-logs.txt` SHA-256: `53fc340931a9d4940bece26ba436a6b329c87bef2fe0cc978f016cf7821ec5de`.

### Publication recovery history

A first `--no-run` save produced revision 6, but the then-current deployment helper expected the CLI phrase `successfully pushed`; official CLI uses `successfully saved without running` for `--no-run`. The save did not execute the notebook and the helper correctly refused to attest the unparsed revision. The parser and API serialization-aware content comparison were fixed and tested in commit `fc0b9e5`. Revision 7 was saved without running; revision 8 was then executed and independently version-pulled, content-checked, and attested. No more than one notebook execution was performed for this release.

## Verification and independent review

The Kaggle v8 run is bound to immutable source commit `fc0b9e5`; subsequent local-only hardening of the deployment helper does not alter the notebook or scientific runtime executed in that version.

- Final full local regression after deployment hardening: **168 passed, 2 skipped, 145 subtests passed** (`.venv/bin/python -m pytest -q`, 78.60s); focused deployment tests pass (25 tests). `git diff --check` passes.
- Fresh-kernel CPU notebook SMOKE: **PASS**, exporting one REINFORCE and one A2C checkpoint with their integrity checks. It was run after the notebook rewrite and before the later deploy-only revision-parser correction; notebook and training sources were unchanged by that correction.
- Code review: independent final reviewers found no issues in attestation binding or filesystem hardening. Attestation creation performs its own exact numeric-version pull for the staged kernel, recomputes local raw/canonical/source hashes, and refuses duplicate-key/non-standard JSON. Detached evidence writes use exclusive/no-follow creation or atomic same-directory replacement; symlink victim regression tests pass.
- Adversarial QA: initially found and reproduced attestation-evidence and symlink-clobber gaps; each was fixed, regression-tested, and independently rechecked. Current scoped adversarial review found no direct fabricated-attestation or output-path clobber path in the reviewed deployment workflow. Earlier provenance, locked-access, and full-training denial checks also pass.
- Architecture review: independently confirmed Kaggle v8 bounded hardware and artifact claims; staging/provenance, bounded DDP, and same-kernel process-resume are accepted for engineering claims. Reviewer independently verified all 7/7 manifest artifacts and ZIP hash. Note: the notebook-produced readiness artifact itself records `release_manifest_sha256: null`; the detached host-side attestation provides the manifest linkage. Overall full-study readiness remains **BLOCKED** as detailed below.
- Secret scan of captured Kaggle evidence found no credential/token/private-key markers. No datasets or training outputs from scientific experiments are included.

## Readiness gates

| Gate | Result | Evidence / limitation |
|---|---|---|
| Immutable source, staged notebook, protocol and acceptance-threshold bindings | **PASS (engineering)** | Release manifest and v8 attestation in the evidence bundle. Candidate protocol is visibly draft/pending; no authority is granted. |
| Historical failed evidence, dependency pins, thresholds preserved | **PASS** | Historical reports and `requirements-training.txt` unchanged; no threshold edits. |
| Kaggle authentication, private metadata, Internet, T4x2 | **PASS** | Existing authenticated CLI; version 8 metadata/API response and run logs. No credentials printed. |
| CUDA forward/backward and both-GPU training work | **PASS (bounded)** | T4 device names, rank/sample counters, finite gradients, parameter deltas and synchronized digests recorded. No acceleration claim. |
| Interrupted-training recovery | **PASS, narrow scope** | New trainer process resumed from committed update 1 within the same Kaggle kernel. **Kaggle session/kernel loss recovery remains unverified.** |
| Exported checkpoint/metrics/manifest integrity | **PASS (readiness artifacts only)** | 7 manifest entries and ZIP SHA verified after retrieval. Not scientific model exports or selected checkpoints. |
| Full matched REINFORCE vs A2C training, held-out evaluation, action-collapse and cognition analyses | **NOT RUN / BLOCKED** | No scientific outcomes authorized or generated. Existing policy-collapse and failed validation evidence remain unchanged. |
| Precision-only pilot, frozen `delta0`, validation checkpoint selection | **NOT RUN / NOT COMPLETE** | Pending authorization; no Torch precision-pilot result was generated. Finalizer accepts masked pilot artifacts and binds `delta0`, but there is no completed Torch pilot in this release. |
| Preregistered multi-checkpoint crossed-bootstrap/Holm locked analysis | **FAIL / BLOCKER** | Current evaluator explicitly states it does not implement this complete analysis. Do not access locked outcomes until the analysis is implemented, independently reviewed, and frozen. |
| Protocol approval and locked access trust root | **BLOCKED BY GOVERNANCE** | Runtime-compatible v2 candidate is separately versioned but remains `DRAFT_REVIEW_REQUIRED_NO_OUTCOMES`; approval flags are false and the pinned trusted-signer allowlist is empty. Do not bypass either guard. |
| Biological validity / husbandry suitability | **OUT OF SCOPE / NOT CLAIMED** | No biological validation was performed. |

**Overall:** this is a verified engineering readiness release, not an experimental release authorized to collect study outcomes. The most immediate blockers are (1) implement and independently review the preregistered pilot/analysis path without changing the frozen criteria; (2) record separate explicit approval for the runtime protocol and a governed trusted signer; and (3) only then run the precision pilot and freeze `delta0`/selection/analysis before any locked access. Architecture review also requires protected durable storage for the append-only locked-access audit if a future locked run is authorized.

## Reproducible Kaggle commands

Run from a clean clone of the GitHub repository. Use the isolated official CLI source snapshot with version-aware `kernels pull`; do not substitute a global CLI that lacks the `--no-run` flag or merged version-selection fix. The runtime profile is historical Kaggle v5 evidence, not a guarantee that a future image is unchanged; stop and review a new profile if it differs.

```bash
export PATH="$HOME/.cache/pasture-kaggle-cli/venv-6894864ff4095cf98e9e36d3a694ff5fc9ba3818/bin:$PATH"
kaggle kernels list --mine --page-size 1  # read-only authentication check
kaggle kernels push --help | grep -- --no-run

cat >/tmp/pasture-kaggle-runtime.json <<'JSON'
{"python":"3.13.15","numpy":"2.1.3","pytorch":"2.11.0+cu128","cuda_runtime":"12.8","cuda_available":true,"gpu_count":2,"gpu_names":["Tesla T4","Tesla T4"]}
JSON

.venv/bin/python scripts/kaggle_deploy.py prepare-release \
  --path /tmp/pasture-kaggle-readiness --owner pascalcoimbra \
  --runtime-profile /tmp/pasture-kaggle-runtime.json
.venv/bin/python scripts/kaggle_release.py generate \
  --runtime-profile /tmp/pasture-kaggle-runtime.json \
  --staged-notebook /tmp/pasture-kaggle-readiness/pasture_studies_kaggle.ipynb \
  --output /tmp/pasture-kaggle-readiness-release-manifest.json
.venv/bin/python scripts/kaggle_deploy.py bind-release \
  --path /tmp/pasture-kaggle-readiness \
  --manifest /tmp/pasture-kaggle-readiness-release-manifest.json
.venv/bin/python scripts/kaggle_release.py gate READINESS \
  /tmp/pasture-kaggle-readiness-release-manifest.json \
  --config /tmp/pasture-kaggle-readiness-readiness-config.json \
  --staged-notebook /tmp/pasture-kaggle-readiness/pasture_studies_kaggle.ipynb

# Publish without execution, then execute only after reviewing all local bindings.
.venv/bin/python scripts/kaggle_deploy.py publish --path /tmp/pasture-kaggle-readiness
.venv/bin/python scripts/kaggle_deploy.py execute \
  --path /tmp/pasture-kaggle-readiness --mode READINESS \
  --confirm-execution --confirm-gpu

# Replace VERSION with the numeric revision emitted by the helper.
.venv/bin/python scripts/kaggle_deploy.py status pascalcoimbra/pasture-studies/VERSION
.venv/bin/python scripts/kaggle_deploy.py logs pascalcoimbra/pasture-studies/VERSION
.venv/bin/python scripts/kaggle_deploy.py output \
  pascalcoimbra/pasture-studies/VERSION --path /tmp/pasture-kaggle-output
.venv/bin/python scripts/kaggle_deploy.py verify \
  /tmp/pasture-kaggle-output/pasture-release-readiness-RUNID/artifact-manifest.json \
  --artifacts /tmp/pasture-kaggle-output/pasture-release-readiness-RUNID
sha256sum /tmp/pasture-kaggle-output/pasture-release-readiness-RUNID/pasture-artifacts.zip
```

Checkpoint recovery is tested by the notebook's first committed DDP update followed by a fresh `torchrun` process resuming `LATEST`. For Kaggle session loss, copy the complete checkpoint directory and manifest into durable storage before terminating the session; cross-session persistence has not been demonstrated. `EXPERIMENT` and `FULL_VALIDATION` are fail-closed and are not part of these commands.
