# Pasture Studies Kaggle Notebook Audit

**Verdict: PASS for the local notebook pipeline smoke and artifact-integrity checks. NOT RUN for live Kaggle/CUDA validation. EXPERIMENT and FULL_VALIDATION are blocked in this publication.** This is not a learning, cognition, biological, or GPU-performance validation report.

## Scope and outcome

The notebook pins its training source to immutable commit `74f5ccd5b6dd785b2e424d29302c15bcec73bbc1`. It defaults to `SMOKE`; `PREFLIGHT` and `SMOKE` are the only enabled modes. `EXPERIMENT` and `FULL_VALIDATION` are hard-disabled because the executing Kaggle notebook bytes do not have independently verified, source-bound approval. Caller-provided revision/hash strings cannot authorize those modes. The notebook records this provenance limitation rather than treating the pinned source notebook hash as an attestation of the executing notebook.

The audit addressed the notebook comparison-cell helpers against the actual paired benchmark schema, removed the duplicate device-report definition, and added dependency/version and CUDA preflight behavior. Preflight preserves the exact dependency-release pins, skips installation when they already match, and does not silently replace an incompatible CUDA environment. The notebook includes a real CUDA forward/backward smoke when usable CUDA is present; this check was not executed on the local CPU-only host.

## Verification evidence

| Check | Result | Evidence / boundary |
| --- | --- | --- |
| Fresh-kernel top-to-bottom notebook smoke | **PASS** | `PASTURE_RUN_NOTEBOOK_SMOKE=1 .venv/bin/python -m unittest discover -s tests -p test_pasture_notebook_smoke.py -v` — 3 passed. Executes all notebook code cells in a fresh kernel with a minimal CPU REINFORCE/A2C workload; verifies pinned source and preregistration artifacts, checkpoints, export manifest, ZIP integrity, and file hashes. |
| Full local regression suite | **PASS** | `.venv/bin/python -m unittest discover -s tests -v` — 133 tests, 2 skipped. Skips include the opt-in notebook smoke when not enabled and the unavailable CUDA check. |
| Notebook helper/provenance tests | **PASS** | 12 tests, 1 opt-in skip; includes comparison schema and mocked `LATEST` resume-selection coverage. The resume assertion mocks training; it does not validate an interrupted Kaggle training session. |
| Whitespace/diff check | **PASS** | `git diff --check` passed. |
| Historical failure evidence | **PRESERVED** | `validation/MVP1_REMEDIATION_REPORT.md` SHA-256 remains `c11b4a04910045e9278ba703223d303c9f4c22f2536f0e0423f2bba5571166ee`. |
| Acceptance criteria and preregistration | **PRESERVED / PENDING** | Existing thresholds were not changed. Preregistration remains `PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0`; it is not a completed learning validation. |
| Live Kaggle execution and CUDA forward/backward smoke | **NOT RUN** | Local PyTorch is `2.14.0+cpu`; CUDA is unavailable. Kaggle hardware/runtime compatibility remains unverified. |
| GPU acceleration, learned cognition, biological validity | **NOT ESTABLISHED** | No corresponding measurements or validation were performed. |

## Limitations and stop condition

The local smoke proves that the notebook can execute its enabled minimal CPU pipeline in a fresh kernel and produce integrity-checked artifacts. It does not validate CUDA execution, Kaggle's installed environment, scientific training, experimental resume, learning acceptance gates, or performance. Although the notebook contains a CUDA model forward/loss/backward/gradient check for a CUDA-capable runtime, no local CUDA run was possible. Experimental modes remain unavailable until a separately reviewed release can verify and approve the exact executing notebook bytes and bind them to the source and unchanged preregistration. Do not interpret smoke checkpoints as trained policy results.

No MVP2 work was started. No claim of improved cognition, GPU acceleration, or biological validity is made.
