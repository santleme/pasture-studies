# Pasture Studies Kaggle training guide

`notebooks/pasture_studies_kaggle.ipynb` provides a Kaggle entry point for the local PyTorch trainer. The NumPy trainer remains the baseline. Keep the historical MVP1 failure evidence and preregistered acceptance thresholds unchanged. This synthetic experiment does not establish biological validity or production-husbandry guidance.

## Choose an execution mode

The source notebook defaults to `SMOKE`; it does not launch the complete experiment automatically. The helper accepts only `PREFLIGHT` and `SMOKE`. Its `prepare --mode ...` command injects the selected mode as a fixed assignment into the staged notebook copy; the repository source notebook is left unchanged. That staging edit and its recorded hash are not proof of remote notebook provenance or a bypass of the notebook's source-bound experimental gates. For direct use in the Kaggle UI, configure `PASTURE_MODE` before running cells. The optional revision and content-hash environment variables accept caller-provided strings for audit display only: neither value is verified against Kaggle's active notebook bytes or treated as trusted publication provenance.

```python
import os
os.environ["PASTURE_MODE"] = "SMOKE"  # PREFLIGHT or SMOKE only
# Optional caller claim only; it is not verified against active Kaggle bytes:
# os.environ["PASTURE_NOTEBOOK_REVISION"] = "<saved Kaggle notebook revision>"
# Optional caller claim only; it is not verified against active Kaggle bytes:
# os.environ["PASTURE_NOTEBOOK_CONTENT_SHA256"] = "<sha256 claim>"
```

| Mode | Behavior |
| --- | --- |
| `PREFLIGHT` | Verifies Python/dependencies, CUDA visibility and runtime, immutable source checkout, and preregistration provenance. Does not train. |
| `SMOKE` (default) | Uses the Kaggle image's already-installed libraries without package installation. Runs one disposable CPU update each for REINFORCE/A2C and, when CUDA works, one CUDA A2C update; exports checkpoints and artifacts. This is not a training arm or learning result. |
| `EXPERIMENT` | **Hard-disabled in this publication.** Do not use for training. |
| `FULL_VALIDATION` | **Hard-disabled in this publication.** Do not use for training or benchmarking. |

Do not infer GPU acceleration from a visible device, CUDA smoke test, or successful training run. Report measured benchmark results and their limitations; an unavailable CUDA run is **not measured**, not a CPU-result substitute.

## Kaggle setup and dependency preflight

1. Open the notebook from the public [`santleme/pasture-studies`](https://github.com/santleme/pasture-studies) repository. In Kaggle settings choose Python and enable Internet for the pinned GitHub source fetch. Choose GPU/T4x2 only when available and needed. Kaggle allocation and quota can vary.
2. The notebook fetches and checks out the full SHA in `PASTURE_PIPELINE_COMMIT` in detached HEAD and aborts on a mismatch. For a reviewed implementation update, publish the source commit first, then update this pin to that exact commit and regenerate the source-bound preregistration/manifests; never mutate the historical acceptance thresholds or reuse an attestation bound to different source bytes.
3. SMOKE records Python/NumPy/PyTorch versions and comparison with the unchanged historical `requirements-training.txt` pins, but deliberately does not install or replace libraries. This allows it to use a compatible preinstalled CUDA-enabled PyTorch wheel (for example, a Kaggle `+cu128` build) without forcing old pins onto a newer Python image. Runtime mismatches are included in the export separately from the scientific pins. `PREFLIGHT` keeps the strict exact-pin check; it does not redefine the scientific environment.
4. If CUDA is requested or detected for SMOKE, PyTorch must import and report a usable CUDA runtime; otherwise the run fails closed. A GPU-staged notebook sets `PASTURE_REQUIRE_CUDA=1`, so it cannot silently continue on CPU. The smoke path never runs pip install. A T4x2 request can still expose a different device count depending on Kaggle allocation; device names/count are recorded.
5. CUDA validation performs a real model forward, loss, backward pass, and finite-gradient check. When usable CUDA is available, the smoke then runs one explicit single-algorithm/single-update A2C CUDA job. CPU REINFORCE and A2C jobs each use the same bounded one-update command. These operations are diagnostic pipeline checks only, not hardware benchmarks or learning claims.

The notebook creates an outcome-blind preregistration from the pinned checkout and records its manifest and threshold hashes. Its notebook hash refers to the older notebook in the pinned source commit; it is **not** an attestation or approval for the revised executing Kaggle notebook. The export labels `PASTURE_NOTEBOOK_REVISION` and `PASTURE_NOTEBOOK_CONTENT_SHA256` values as unverified caller claims. The notebook cannot inspect or validate the active Kaggle notebook bytes from those strings, and they do not establish publication provenance, approval, or experiment eligibility. `EXPERIMENT` and `FULL_VALIDATION` are hard-disabled in this publication regardless of user-supplied environment variables or JSON/configuration data; caller-provided data is not authorization. Use only `PREFLIGHT` or `SMOKE`. Those experimental modes require a future trusted, source-bound release that independently verifies and approves the exact executing notebook bytes and their source-bound provenance. Until that release exists, do not treat this notebook as eligible for experimental execution; do not reuse an existing run ID or claim an experiment is eligible.

## Run and retrieve smoke artifacts

`PASTURE_RUN_ID` is an optional safe prefix; each SMOKE kernel execution appends a fresh random suffix and creates a new output directory, so it cannot accept prior checkpoints or metrics. The smoke runner also fails if any target output already exists and requires exactly one newly created checkpoint generation plus both inference exports. SMOKE checkpoints are disposable pipeline-test artifacts, not experimental training checkpoints. `--smoke-update` requires exactly one algorithm and explicit CPU/CUDA, performs only one complete batch/update, disallows resume and distributed runs, and writes a checkpoint plus NumPy inference export. Experimental Kaggle resume is unavailable while `EXPERIMENT` and `FULL_VALIDATION` remain hard-disabled.

`/kaggle/working` may not survive a Kaggle session reset. Before stopping or resetting a session, download the exported SMOKE `pasture-training-artifacts.zip` and `artifact_manifest.json` from notebook output if you need those diagnostic artifacts. Restoring an archive to resume experimental training is a future workflow only; it is not available in this publication. A future trusted, approved source-bound release must verify that resume behavior and document its exact recovery procedure before users rely on it.

After execution, download `export/pasture-training-artifacts.zip` and `export/artifact_manifest.json` from Kaggle's notebook output. The manifest records source commit, run ID, mode, dependencies/runtime, the supplied notebook revision/content hash as **unverified caller claims**, preregistration/threshold hashes, and SHA-256/size for each exported run artifact. Those claims do not validate the active Kaggle notebook bytes. Keep the manifest with the archive. Verify the hashes locally before using a checkpoint. For local cattle-simulation inference, use a checkpoint generation's NumPy `.npz` inference export with its matching JSON sidecar; optimizer/RNG state is for training resume, not inference.

## Kaggle CLI preparation, publication, and explicit execution

The helper stages only `PREFLIGHT` or `SMOKE`, by inserting the requested fixed mode into a copy of the source notebook. Preparation is local and does not execute it:

```bash
python3 scripts/kaggle_deploy.py preflight
# auth_status is not_verified; authenticate manually if needed, then validate separately:
kaggle kernels list --mine --page-size 1
python3 scripts/kaggle_deploy.py prepare --path /tmp/pasture-kaggle-upload --owner <your-kaggle-username> --mode SMOKE
```

To request Kaggle T4×2 metadata, add `--gpu` to `prepare`; this only sets `enable_gpu` and the `NvidiaTeslaT4` machine shape. It does not allocate hardware or execute. The helper's `preflight` deliberately does not authenticate or inspect credentials; use `kaggle auth login` yourself when appropriate, then the read-only `kernels list` check above.

Publication is strictly no-run and fails closed unless the installed CLI advertises `--no-run`:

```bash
python3 scripts/kaggle_deploy.py publish --path /tmp/pasture-kaggle-upload
```

The installed Kaggle CLI 2.2.4 does not list `--no-run` in `kaggle kernels push --help`; **do not publish with that version** because the ordinary push may execute the notebook. Upgrade and verify the active executable's help before attempting a no-run publication. Never remove the safety flag or fall back to an ordinary push.

Running a staged notebook is a separate, explicit action that requires `--confirm-execution` and a matching mode. For a deliberate CPU smoke:

```bash
python3 scripts/kaggle_deploy.py execute --path /tmp/pasture-kaggle-upload --mode SMOKE --confirm-execution
```

For a T4×2 staging folder prepared with `--gpu`, execution additionally requires `--confirm-gpu`, and the helper passes `--accelerator NvidiaTeslaT4`:

```bash
python3 scripts/kaggle_deploy.py execute --path /tmp/pasture-kaggle-upload --mode SMOKE --confirm-execution --confirm-gpu
```

After publication, create a provenance record using the exact Kaggle revision identifier (not the local CLI version):

```bash
python3 scripts/kaggle_deploy.py record ./deployment-record.json \
  --kernel <owner>/<kernel-slug> \
  --source-commit "$(git rev-parse HEAD)" \
  --kernel-version <exact-kaggle-revision> \
  --mode SMOKE \
  --metadata /tmp/pasture-kaggle-upload/kernel-metadata.json
```

The record distinguishes `kaggle_kernel_version` and `kaggle_cli_version`, and stores SHA-256 values for the staged notebook and metadata. Use version-qualified `owner/slug/<numeric-version>` references for status/logs/output when retrieving a specific revision; see the [Kaggle CLI deployment guide](kaggle-cli-deployment.md) for detailed commands and limitations.

No Kaggle authentication, publication, execution, or GPU operation was performed while updating these docs. Neither staging nor publication bypasses the source notebook's disabled `EXPERIMENT`/`FULL_VALIDATION` gates or establishes trusted provenance.

## Local notebook validation

Run the fresh-kernel test from the repository root after installing the optional test requirements. It clones the immutable source pin and executes the notebook top to bottom in `SMOKE` mode with a temporary work directory and CUDA disabled; it verifies checkpoint/export contents and archive integrity. It does not run experimental arms or measure GPU performance.

```bash
python -m pip install -r requirements-notebook-test.txt
PASTURE_RUN_NOTEBOOK_SMOKE=1 python -m unittest discover -s tests \
  -p test_pasture_notebook_smoke.py -v
```

The test requires network access to clone the pinned public source. A CPU-only pass validates notebook setup, smoke training, resume selection, and artifact export—not CUDA execution, learning acceptance, or acceleration. Kaggle CUDA/T4 and performance benchmarks remain unverified until run on the specified Kaggle hardware.

## Frozen interpretation boundaries

Preserve the approved acceptance thresholds and the failed MVP1 evidence in `validation/MVP1_REMEDIATION_REPORT.md`. The fixed-corpus gate remains: GPU-vs-best-CPU throughput 95% CI lower bound **≥2.0×** and elapsed-time reduction **≥30%**; T4x2 must additionally reach **≥1.20×** vs. T4x1. The dynamic end-to-end gate remains at least five paired complete runs, 95% CI lower speedup **≥2×**, elapsed reduction **≥30%** vs. CPU (T4x2 also **≥1.20×** vs. T4x1), exact update/transition counts, action disagreement **≤0.5%**, and each named simulator-work counter within **1.0%**. Apply the source-bound preregistration for all other criteria; do not change thresholds after observing outcomes.

The preregistration remains `PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0`; the precision-only pilot and locked-test binding have not been established. Validation diagnostics do not constitute a locked-test result or prove cognition. A GPU smoke or device visibility does not constitute a benchmark. No result from this notebook alone establishes biological validity.

References: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks), [Kaggle Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage), [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html), and [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
