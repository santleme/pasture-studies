# Pasture Studies Kaggle release workflow

## Execution boundary

Local execution is limited to development, lightweight tests, and explicitly
bounded smoke diagnostics. Full training (including DAgger, PPO, and A2C), mass
trajectory generation, and extensive benchmarks must execute on Kaggle rather
than a local workstation. This is an execution-location requirement, not an
authorization: the current launcher supports bounded PREFLIGHT/SMOKE/READINESS
paths only, and EXPERIMENT/FULL_VALIDATION remain fail-closed pending separately
trusted approval. Do not use local CLI flags or environment variables to bypass
that release gate. Do not run Kaggle jobs or upload artifacts without an
independent approved release.

The Kaggle entry point is a source-bound launcher, not the training engine. Training and checkpoint semantics live in `validation/train_torch_mvp1.py` and `farm_sim/`; scientific protocol lives in the preserved preregistration and historical reports. `notebooks/pasture_studies_kaggle.ipynb` offers only `PREFLIGHT`, disposable `SMOKE`, and bounded `READINESS` paths. `EXPERIMENT` and `FULL_VALIDATION` are fail-closed unless a future independently trusted release approval exists. The current release API has no trust root; caller-supplied environment variables, JSON, notebook revision strings, or files cannot authorize these modes.

No new scientific outcome is collected by release preparation or publishing. READINESS is a non-scientific T4x2 pipeline diagnostic only: two DDP updates, 32 diagnostic episodes, checkpoint recovery between process invocations, both GPUs participating, no evaluation, no preregistration seed splits, no pilot/locked outcome access, and no eligibility for speed, learning, or policy-promotion claims. Historical MVP1 failure evidence, reward definitions, acceptance criteria, thresholds, and dependency pins are unchanged. The Kaggle runtime is recorded separately; it is not silently converted into the formal scientific dependency environment.

## Release manifest and staging

Create an exact runtime profile from the target Kaggle runtime you intend to release against. Example based on the observed Kaggle v5 runtime; if Kaggle's image changes, stop and produce a separately reviewed/versioned runtime profile rather than modifying pins or claiming equivalence:

```json
{
  "python": "3.13.15",
  "numpy": "2.1.3",
  "pytorch": "2.11.0+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "gpu_count": 2,
  "gpu_names": ["Tesla T4", "Tesla T4"]
}
```

The profile is an expected runtime binding. The notebook verifies it against actual Python, NumPy, PyTorch, CUDA, and GPU names before READINESS. It does not install dependencies. Do not use this Kaggle profile as a substitute for the original preregistered scientific environment.

Run from a clean source checkout. The release manifest is intentionally external to the repository because it binds the source commit; committing it into that same source would make the commit hash self-referential. It binds source SHA, exact engine file hashes, exact notebook byte/content hashes, preregistration and threshold hashes, seed/config declarations, and runtime profile. Its status is pending approval and it does **not** authorize EXPERIMENT/FULL_VALIDATION.

```bash
# Existing Kaggle authentication, read-only check; no credentials are printed.
# Use the isolated official CLI snapshot that contains version-aware pull (not a
# globally installed 2.2.4 package lacking the merged PR #1201 fix).
export PATH="$HOME/.cache/pasture-kaggle-cli/venv-6894864ff4095cf98e9e36d3a694ff5fc9ba3818/bin:$PATH"
kaggle --version
kaggle kernels list --mine --page-size 1

cat >/tmp/pasture-kaggle-runtime.json <<'JSON'
{"python":"3.13.15","numpy":"2.1.3","pytorch":"2.11.0+cu128","cuda_runtime":"12.8","cuda_available":true,"gpu_count":2,"gpu_names":["Tesla T4","Tesla T4"]}
JSON

python3 scripts/kaggle_deploy.py prepare-release \
  --path /tmp/pasture-kaggle-readiness \
  --owner <kaggle-username> \
  --runtime-profile /tmp/pasture-kaggle-runtime.json

# Manifest generation is after staging so it binds the exact upload bytes.
python3 scripts/kaggle_release.py generate \
  --runtime-profile /tmp/pasture-kaggle-runtime.json \
  --staged-notebook /tmp/pasture-kaggle-readiness/pasture_studies_kaggle.ipynb \
  --output /tmp/pasture-release-manifest.json
python3 scripts/kaggle_deploy.py bind-release \
  --path /tmp/pasture-kaggle-readiness \
  --manifest /tmp/pasture-release-manifest.json
python3 scripts/kaggle_release.py verify /tmp/pasture-release-manifest.json \
  --staged-notebook /tmp/pasture-kaggle-readiness/pasture_studies_kaggle.ipynb
python3 scripts/kaggle_release.py gate READINESS /tmp/pasture-release-manifest.json \
  --config /tmp/pasture-kaggle-readiness-readiness-config.json \
  --staged-notebook /tmp/pasture-kaggle-readiness/pasture_studies_kaggle.ipynb
```

`prepare-release` checks the target profile is CUDA-enabled T4x2, then stages an injected notebook copy and private `kernel-metadata.json` with Internet + `NvidiaTeslaT4`. Kaggle CLI uploads only the `code_file` notebook; it does not upload sidecar JSON files. Therefore the immutable safe deployment config is embedded as a literal in the staged notebook, while the deployment record and release manifest remain detached host-side evidence. The manifest is generated after staging and binds both committed source notebook bytes and exact staged notebook bytes/content. `bind-release` verifies that exact staged notebook and stores the manifest detached; it does not edit notebook bytes. Staging contains exactly the notebook and metadata and rejects extras. No self-hash is placed in the notebook.

## Publish and verify exact Kaggle revision

The publish helper requires `kaggle kernels push --no-run` to appear in the active CLI's help. It parses the server-assigned numeric revision from either official save response (`Kernel version N successfully saved without running`) or executing-push response, requests that exact numeric version using `kernels pull owner/slug/N`, and compares canonical full notebook content and source cells. The attestation records both local/remote raw notebook SHA-256 and their canonical content SHA-256; Kaggle may serialize `cell.source` as a string rather than a line list, so byte hashes can differ while the entire parsed notebook content remains identical after that representation-only normalization. Do not use the original 2.2.4 code that treated the suffix as part of the slug; use the isolated official Kaggle CLI source snapshot that includes merged PR #1201 (commit `6894864ff4095cf98e9e36d3a694ff5fc9ba3818` in this environment), where pull sends `version_label=vN`. The merged upstream change reports live API validation for version-specific pull. The helper never infers a revision from a latest listing or ambiguous `latest` selector. It writes a detached, credential-free `*-publication-attestation.json` only after version-specific content comparison passes.

```bash
python3 scripts/kaggle_deploy.py preflight
python3 scripts/kaggle_deploy.py publish --path /tmp/pasture-kaggle-readiness
# Output identifies the numeric revision printed by Kaggle and the detached attestation path.
python3 scripts/kaggle_deploy.py compare \
  --path /tmp/pasture-kaggle-readiness \
  --kernel <kaggle-username>/pasture-studies --version <numeric-revision>
```

If your CLI does not advertise `--no-run`, do not publish with ordinary push as a workaround. Upgrade/configure the official CLI and recheck help. Keep the release manifest and deployment files private.

## Explicit bounded READINESS run

Execution uses the existing deployment helper. This command consumes Kaggle GPU quota and should only be used after the READINESS release has completed independent code/architecture/adversarial reviews. It cannot run EXPERIMENT or FULL_VALIDATION:

```bash
python3 scripts/kaggle_deploy.py execute \
  --path /tmp/pasture-kaggle-readiness --mode READINESS \
  --confirm-execution --confirm-gpu
```

The notebook checks the exact pinned source checkout and embedded runtime/config bindings, fails unless two T4 CUDA devices are visible, performs a real forward/backward gradient check, then runs bounded A2C DDP update 1 followed by a fresh torchrun invocation resuming from `LATEST` for update 2. It validates rank/device participation, finite gradients, synchronized parameter digests, checkpoint generations, and readiness-only flags. It performs no policy evaluation. Wall-clock output is diagnostic only and is not a hardware-performance benchmark.

Monitor and retrieve using the exact numeric version printed by `execute` or `publish`:

```bash
python3 scripts/kaggle_deploy.py status <kaggle-username>/pasture-studies/<numeric-revision>
python3 scripts/kaggle_deploy.py logs <kaggle-username>/pasture-studies/<numeric-revision>
python3 scripts/kaggle_deploy.py output <kaggle-username>/pasture-studies/<numeric-revision> \
  --path /tmp/pasture-kaggle-output
python3 scripts/kaggle_deploy.py verify \
  /tmp/pasture-kaggle-output/pasture-release-<run-id>/artifact-manifest.json \
  --artifacts /tmp/pasture-kaggle-output/pasture-release-<run-id>
sha256sum /tmp/pasture-kaggle-output/pasture-release-<run-id>/pasture-readiness.zip
```

The notebook writes checkpoint generations, readiness evidence, observed runtime, ZIP, and SHA-256 manifest under its fresh `/kaggle/working/pasture-release-readiness-<random-id>/` directory. Keep all files together. A readiness checkpoint cannot be promoted or treated as a scientific training checkpoint. For interruption recovery testing, the notebook intentionally stops after its first committed update and starts a new torchrun invocation with `--resume latest`; this tests actual checkpoint deserialization/restoration in the same Kaggle kernel session. It does not claim survival after Kaggle destroys `/kaggle/working`.

## Historical gates and limitations

- Historical requirements remain in `requirements-training.txt`. Kaggle's Python 3.13/NumPy 2/PyTorch CUDA stack is a separately bound readiness runtime, not a rewritten protocol.
- No LoRA/QLoRA, external LLM, hosted inference, or model weights were added.
- A visible GPU or successful readiness step is not GPU acceleration evidence. No throughput comparison is executed in READINESS.
- Pending release status is not approval. Do not access pilot or locked-test outcomes or launch full-scale training without a separate authorized, independently verifiable release.
- No claim of biological validation, cognition acceptance, or production husbandry suitability.

The previous successful Kaggle SMOKE v5 record remains in `validation/KAGGLE_SMOKE_V5_REPORT.md`; it is historical evidence and is not overwritten by this launcher release.

## User-authorized exploratory A2C run (nonformal)

`EXPLORATORY` is a separate, fixed-budget path for hands-on exploration—not a way to unlock `EXPERIMENT` or `FULL_VALIDATION`. It trains A2C for exactly 256 complete 72-tick episodes (16 optimizer updates) on one T4 from the Kaggle T4x2 profile, checkpoints every update, records reward/action/invalid-action metrics, exports a local-inference `.npz` policy, and runs five generated exploratory worlds with all five animals for behavior inspection. The fixed seed is `9482817`; it is used only as an unregistered exploratory seed. Outputs are tagged `exploratory_only=true`, `formal_eligible=false`, and `benchmark_eligible=false`. No preregistered splits, pilot, held-out, locked outcomes, selection rules, or scientific acceptance thresholds are used. The resulting policy is not promoted into the app's default policy automatically.

After a clean commit is pushed to the public source repository, use the same existing deployment workflow:

```bash
python3 scripts/kaggle_deploy.py prepare-exploratory \
  --path /tmp/pasture-kaggle-exploratory \
  --owner pascalcoimbra \
  --runtime-profile /tmp/pasture-kaggle-runtime.json
python3 scripts/kaggle_release.py generate \
  --runtime-profile /tmp/pasture-kaggle-runtime.json \
  --staged-notebook /tmp/pasture-kaggle-exploratory/pasture_studies_kaggle.ipynb \
  --output /tmp/pasture-exploratory-release.json
python3 scripts/kaggle_release.py gate EXPLORATORY /tmp/pasture-exploratory-release.json \
  --config /tmp/pasture-kaggle-exploratory-config.json \
  --staged-notebook /tmp/pasture-kaggle-exploratory/pasture_studies_kaggle.ipynb
python3 scripts/kaggle_deploy.py bind-release \
  --path /tmp/pasture-kaggle-exploratory --manifest /tmp/pasture-exploratory-release.json
python3 scripts/kaggle_deploy.py publish --path /tmp/pasture-kaggle-exploratory
python3 scripts/kaggle_deploy.py compare \
  --path /tmp/pasture-kaggle-exploratory \
  --kernel pascalcoimbra/pasture-studies --version <published-version>
python3 scripts/kaggle_deploy.py execute \
  --path /tmp/pasture-kaggle-exploratory --mode EXPLORATORY \
  --confirm-execution --confirm-gpu
python3 scripts/kaggle_deploy.py status pascalcoimbra/pasture-studies/<executed-version>
python3 scripts/kaggle_deploy.py logs pascalcoimbra/pasture-studies/<executed-version>
python3 scripts/kaggle_deploy.py output pascalcoimbra/pasture-studies/<executed-version> \
  --path /tmp/pasture-kaggle-exploratory-output
python3 scripts/kaggle_deploy.py verify \
  /tmp/pasture-kaggle-exploratory-output/pasture-release-<run-id>/artifact-manifest.json \
  --artifacts /tmp/pasture-kaggle-exploratory-output/pasture-release-<run-id>
# Optional: inspect the downloaded policy in the local 2D app (does not replace the default policy).
python3 -m farm_sim.server \
  --checkpoint /tmp/pasture-kaggle-exploratory-output/pasture-release-<run-id>/exploratory-only/exploratory-only/a2c/torch-a2c-seed-9482817.npz
# Open http://127.0.0.1:8000 and inspect the five per-animal trajectories/action histories.
```

This real training run consumes Kaggle GPU quota. Reward trajectories, action distributions, and rollout traces are descriptive only—not evidence of learning efficacy, cognition, acceleration, or biological validity. No Kaggle execution is initiated by documentation preparation itself.

## Exploratory A2C comparison (A–D, one 12-job batch)

`EXPLORATORY_COMPARISON` is separate from the formal protocol. It trains exactly 12 A2C policies from scratch: A (original reward/objective), B (original reward with normalized advantages and entropy coefficient 0.01), C (state-aware exploratory reward with the original objective), and D (state-aware reward plus the B objective); each condition uses the same three fixed training seeds and 512 complete 72-tick episodes per policy. The optional reward shape is scale 0.1, only used by C/D; its A2C returns remain undiscounted (`discount_factor=1.0`). Evaluations use one fixed, independent evaluation root seed and the original simulator reward on common synthetic scenario worlds, and compare every checkpoint with same-seed untrained and heuristic baselines. The comparison reports rewards, resource events, final toy drives, action frequencies/entropy, invalid actions, and first-decision memory/perception probes. A descriptive ranking and SVG are not acceptance or biological claims.

Before those 12 jobs, the notebook performs a bounded no-optimizer/no-checkpoint calibration on CPU, each T4, and two concurrent one-batch T4 workers. Only one training schedule is selected: two independent GPU shards if their measured throughput exceeds the best single-device throughput, otherwise one fastest device (including CPU if it actually wins). This is a performance diagnostic, not a learning run or GPU speedup claim. The current execution must run only one 12-job suite. The 24-job ceiling reserves at most one additional 12-job exploratory iteration after reviewing the complete first-batch evidence; it is not a duplicate single-GPU/dual-GPU comparison. No model is selected using formal or locked data; no preregistration thresholds are changed.

After pushing the source commit to `origin`, deploy and execute using the existing CLI workflow:

```bash
python3 scripts/kaggle_deploy.py preflight
python3 scripts/kaggle_deploy.py prepare-exploratory-comparison \
  --path /tmp/pasture-kaggle-a2c-comparison \
  --owner pascalcoimbra \
  --runtime-profile validation/kaggle-exploratory/kaggle-v11-runtime-profile.json
python3 scripts/kaggle_deploy.py execute \
  --path /tmp/pasture-kaggle-a2c-comparison \
  --mode EXPLORATORY_COMPARISON --confirm-execution --confirm-gpu
```

Record the exact numeric notebook version reported by the push, then monitor and retrieve that exact version (do not use `latest`):

```bash
python3 scripts/kaggle_deploy.py status pascalcoimbra/pasture-studies/<version>
python3 scripts/kaggle_deploy.py logs pascalcoimbra/pasture-studies/<version>
python3 scripts/kaggle_deploy.py output pascalcoimbra/pasture-studies/<version> \
  --path /tmp/pasture-kaggle-a2c-output
```

The notebook artifact manifest binds the source commit, notebook hash, runtime, calibration, all 12 training jobs, full checkpoints, local `.npz`/sidecar inference exports, raw metric curves, and common-world JSON/SVG evaluation. Verify the downloaded archive and its manifest before loading any policy:

```bash
python3 scripts/kaggle_deploy.py verify \
  /tmp/pasture-kaggle-a2c-output/pasture-release-<run-id>/artifact-manifest.json \
  --artifacts /tmp/pasture-kaggle-a2c-output/pasture-release-<run-id>
```

The exported `torch-a2c-seed-<seed>.npz` plus same-basename `.json` file can be loaded by the local simulation without promotion or replacing the default policy:

```bash
python3 -m farm_sim.server --checkpoint \
  /tmp/pasture-kaggle-a2c-output/pasture-release-<run-id>/exploratory-comparison/training-suite/exploratory-comparison/variant-<A-D>/seed-<seed>/a2c/torch-a2c-seed-<seed>.npz
# Open http://127.0.0.1:8000 and inspect the five simulated animals.
```

If A-D evaluation confirms reward exploitation (specifically no forage successes with near-maximal final hunger), the bounded follow-up is a separate `EXPLORATORY_ITERATION_2` mode: E/F each use three new seeds and 512 episodes, state-aware reward scale `0.5` (up from `0.1`), and the same original-vs-normalized/entropy A2C objective contrast. Its six jobs bring the cumulative count to 18/24; it does not change the original reward, the formal protocol, or its acceptance criteria. Stage and run only after inspecting the first comparison:

```bash
python3 scripts/kaggle_deploy.py prepare-exploratory-iteration-2 \
  --path /tmp/pasture-kaggle-a2c-iteration-2 \
  --owner pascalcoimbra \
  --runtime-profile validation/kaggle-exploratory/kaggle-v11-runtime-profile.json
# Generate a fresh outcome-blind release manifest with scripts/kaggle_release.py,
# bind it with scripts/kaggle_deploy.py bind-release, then:
python3 scripts/kaggle_deploy.py execute \
  --path /tmp/pasture-kaggle-a2c-iteration-2 \
  --mode EXPLORATORY_ITERATION_2 --confirm-execution --confirm-gpu
```

Iteration-2 outputs are E/F-only; create a combined A-F checkpoint manifest from the verified A-D and E/F final generations and pass it to `validation.compare_exploratory_checkpoints` with the same `9483999` evaluation root to produce a fully common-world six-condition report. Never copy any formal/locked artifacts into this exploratory directory.

If a run is interrupted, the trainer commits a checkpoint generation after each update and resumes from the per-job `LATEST` pointer when the same output directory is rerun. Kaggle output retrieval and the run manifest must be checked before attempting recovery. Count every started policy attempt against the 24-job ceiling. If the failed kernel does not expose its `/kaggle/working` checkpoint outputs, do not claim cross-session resume; report the external limitation and avoid re-running the entire suite if it would exceed the cap.
