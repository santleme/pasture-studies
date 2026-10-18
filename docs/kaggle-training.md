# Pasture Studies Kaggle training guide

`notebooks/pasture_studies_kaggle.ipynb` provides a Kaggle entry point for the local PyTorch trainer. The NumPy trainer remains the baseline. Keep the historical MVP1 failure evidence and preregistered acceptance thresholds unchanged. This synthetic experiment does not establish biological validity or production-husbandry guidance.

## Choose an execution mode

The notebook defaults to `SMOKE`; it does not launch the complete experiment automatically. Set `PASTURE_MODE` before running the notebook top to bottom. In Kaggle, add a short code cell above the notebook's setup cells, set the environment variables there, and use **Run All**. This published revision supports only `PREFLIGHT` and `SMOKE`. The optional revision and content-hash environment variables accept caller-provided strings for audit display only: neither value is verified against Kaggle's active notebook bytes or treated as trusted publication provenance.

```python
import os
os.environ["PASTURE_MODE"] = "SMOKE"  # PREFLIGHT, SMOKE, EXPERIMENT, or FULL_VALIDATION
# Optional caller claim only; it is not verified against active Kaggle bytes:
# os.environ["PASTURE_NOTEBOOK_REVISION"] = "<saved Kaggle notebook revision>"
# Optional caller claim only; it is not verified against active Kaggle bytes:
# os.environ["PASTURE_NOTEBOOK_CONTENT_SHA256"] = "<sha256 claim>"
```

| Mode | Behavior |
| --- | --- |
| `PREFLIGHT` | Verifies Python/dependencies, CUDA visibility and runtime, immutable source checkout, and preregistration provenance. Does not train. |
| `SMOKE` (default) | Runs disposable, minimal CPU REINFORCE and A2C checkpoint checks; if CUDA works, also runs a small CUDA A2C check. Exports artifacts. This is not a training arm or learning result. |
| `EXPERIMENT` | **Hard-disabled in this publication.** Do not use for training. |
| `FULL_VALIDATION` | **Hard-disabled in this publication.** Do not use for training or benchmarking. |

Do not infer GPU acceleration from a visible device, CUDA smoke test, or successful training run. Report measured benchmark results and their limitations; an unavailable CUDA run is **not measured**, not a CPU-result substitute.

## Kaggle setup and dependency preflight

1. Open the notebook from the public [`santleme/pasture-studies`](https://github.com/santleme/pasture-studies) repository. In Kaggle settings choose Python and enable Internet for the pinned GitHub source fetch. Choose GPU/T4x2 only when available and needed. Kaggle allocation and quota can vary.
2. Keep `PASTURE_PIPELINE_COMMIT` at `74f5ccd5b6dd785b2e424d29302c15bcec73bbc1`. The notebook fetches and checks out that exact commit in detached HEAD and aborts on a mismatch. Do not change the source pin for confirmatory runs.
3. Run all cells in order. The notebook parses exact `name==version` pins from the pinned checkout's `requirements-training.txt`, accepts only the pinned release version (ignoring a local wheel build suffix such as `+cu...`), and skips pip installation when they already match. It runs `pip check` and verifies the versions again.
4. If a non-CUDA dependency is missing or mismatched, the notebook installs the pinned requirements and then stops with a kernel-restart instruction. Restart the Kaggle kernel and run all cells again. If CUDA hardware/runtime is expected but the installed PyTorch version mismatches, or the pinned build lacks CUDA support, it fails rather than silently replacing a working CUDA install or falling back to CPU. Resolve the CUDA-compatible pinned environment explicitly; not every Kaggle image is guaranteed to satisfy this exact-version contract.
5. The CUDA check performs a real model forward, loss, backward pass, and finite-gradient check when CUDA is available. If no CUDA device is available, it records the CUDA smoke as skipped unless `PASTURE_REQUIRE_CUDA=1` or NVIDIA hardware is detected; a skip is not a CUDA pass.

The notebook creates an outcome-blind preregistration from the pinned checkout and records its manifest and threshold hashes. Its notebook hash refers to the older notebook in the pinned source commit; it is **not** an attestation or approval for the revised executing Kaggle notebook. The export labels `PASTURE_NOTEBOOK_REVISION` and `PASTURE_NOTEBOOK_CONTENT_SHA256` values as unverified caller claims. The notebook cannot inspect or validate the active Kaggle notebook bytes from those strings, and they do not establish publication provenance, approval, or experiment eligibility. `EXPERIMENT` and `FULL_VALIDATION` are hard-disabled in this publication regardless of user-supplied environment variables or JSON/configuration data; caller-provided data is not authorization. Use only `PREFLIGHT` or `SMOKE`. Those experimental modes require a future trusted, source-bound release that independently verifies and approves the exact executing notebook bytes and their source-bound provenance. Until that release exists, do not treat this notebook as eligible for experimental execution; do not reuse an existing run ID or claim an experiment is eligible.

## Run and retrieve smoke artifacts

The default run ID is `mvp1-run`; its output directory is `/kaggle/working/pasture-training-artifacts-mvp1-run`. Set `PASTURE_RUN_ID` before execution to isolate another run. Currently this only applies to `PREFLIGHT`/`SMOKE`; SMOKE checkpoints are disposable pipeline-test artifacts, not experimental training checkpoints. The trainer's `LATEST` resume-selection logic is covered by a mocked local test, but experimental Kaggle resume is unavailable while `EXPERIMENT` and `FULL_VALIDATION` remain hard-disabled.

`/kaggle/working` may not survive a Kaggle session reset. Before stopping or resetting a session, download the exported SMOKE `pasture-training-artifacts.zip` and `artifact_manifest.json` from notebook output if you need those diagnostic artifacts. Restoring an archive to resume experimental training is a future workflow only; it is not available in this publication. A future trusted, approved source-bound release must verify that resume behavior and document its exact recovery procedure before users rely on it.

After execution, download `export/pasture-training-artifacts.zip` and `export/artifact_manifest.json` from Kaggle's notebook output. The manifest records source commit, run ID, mode, dependencies/runtime, the supplied notebook revision/content hash as **unverified caller claims**, preregistration/threshold hashes, and SHA-256/size for each exported run artifact. Those claims do not validate the active Kaggle notebook bytes. Keep the manifest with the archive. Verify the hashes locally before using a checkpoint. For local cattle-simulation inference, use a checkpoint generation's NumPy `.npz` inference export with its matching JSON sidecar; optimizer/RNG state is for training resume, not inference.

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
