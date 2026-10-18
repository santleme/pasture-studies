# Pasture Studies Kaggle SMOKE v5 — execution and artifact validation

**Verdict: PASS for the authorized SMOKE software/integration gates.** This is not evidence of learning quality, a GPU performance benchmark, biological validation, or a pass of any MVP1 cognition acceptance gate.

## Published execution

- Kaggle notebook: <https://www.kaggle.com/code/pascalcoimbra/pasture-studies>
- Private kernel version **5** completed with `KernelWorkerStatus.COMPLETE`. Version 4 was a no-run save; version 5 was the actual execution.
- Kaggle execution source commit (immutable pipeline checkout): `42a4387ab65553c734a63411b277d0b842ca8660`.
- Published notebook came from repository revision `73a949d5e5970465c0581a377e855a5aee64cb0a`. The staged notebook SHA-256 is `e461d8ebb09eb075b5d7cd9281d3b998f39d519d7422f4104b22574e976c8e3f`; metadata SHA-256 is `5d9c5c8462c2f1ce351e7450487ab5dda822bdb13238650784bb958aba399b51`.
- The exact version-5 notebook was pulled from Kaggle and compared cell-by-cell with staging: all 32 cell sources matched. Retrieved metadata confirms private=true, Internet=true, GPU=true, `NvidiaTeslaT4` (T4×2).
- The Kaggle CLI in the isolated official CLI environment was version 2.2.4 and advertised `kernels push --no-run`. Existing authentication was verified with `kaggle kernels list --mine`; no credentials were printed, changed, or committed.
- The previous failed v2 run is counted as attempt 1; v5 is execution attempt 2 of the six-attempt limit.

## Runtime and SMOKE results

Kaggle reported Python **3.13.15**, NumPy **2.1.3**, PyTorch **2.11.0+cu128**, and CUDA build **12.8**. It exposed **2 × Tesla T4** devices. PyTorch reported CUDA available, and the notebook's real model forward/loss/backward plus finite-gradient check logged **`CUDA forward/backward smoke: PASS Tesla T4`**.

No package installation occurred in SMOKE. The run recorded mismatches against the preserved scientific pins (NumPy 1.26.4 and Torch 2.14.0); it used the preinstalled Kaggle runtime instead. This is justified by the Python compatibility split: NumPy 1.26.4's official notes list Python 3.9–3.12, while NumPy 2.1.3 supports Python 3.13. The preregistered scientific dependency pins and acceptance thresholds were not edited. See [NumPy 1.26.4 notes](https://numpy.org/doc/2.3/release/1.26.4-notes.html), [NumPy 2.1.3 notes](https://numpy.org/doc/stable/release/2.1.3-notes.html), [PyTorch previous-version CUDA wheels](https://pytorch.org/get-started/previous-versions/), [Kaggle Notebook documentation](https://www.kaggle.com/docs/notebooks), and the [official Kaggle CLI](https://github.com/Kaggle/kaggle-cli).

All three bounded smoke jobs completed exactly one update over 16 episodes and produced checkpoint generation 1 plus inference `.npz`/JSON exports:

| Job | Requested / recorded device | Result |
| --- | --- | --- |
| REINFORCE | CPU / CPU | PASS |
| A2C | CPU / CPU | PASS |
| A2C | CUDA / CUDA | PASS |

Each job's metrics record says `update=1`, `episodes_in_batch=16`, and `benchmark_eligible=false`. No full experiment, fixed-corpus benchmark, or full validation ran. The notebook retained `EXPERIMENT` and `FULL_VALIDATION` disabled. The runtime log spans elapsed events from 7.64s through 67.12s; this is log-relative notebook time, not a controlled performance benchmark.

## Download and integrity verification

Downloaded from version 5 through `scripts/kaggle_deploy.py output` into `/tmp/pasture-kaggle-v5-output/`:

- Archive: `pasture-training-artifacts.zip`, **142,182 bytes**, SHA-256 `01557638220f4f67816fef1b3b7e05c410c22d20f06ecfa1abae0fc8d2680bd2`.
- SHA-256 manifest: `artifact_manifest.json`, **6,824 bytes**, SHA-256 `9086268fc67de9c982b2d4c0b97a69dd37f4c3befb06370508bff8138824c80f`.
- The deployment helper verified all **27** manifest entries against the downloaded files. ZIP CRC validation passed; all 27 manifest-listed file paths were present inside the archive and matched their recorded size and SHA-256.
- Per-job `LATEST` pointers resolve to the sole generation-1 checkpoint. Both inference files match their checkpoint-local export-manifest hashes. CPU and CUDA A2C/REINFORCE checkpoint and evaluation records were present.
- The raw Kaggle log and outer artifact manifest are preserved next to this report as `kaggle-smoke-v5-logs.json` and `kaggle-smoke-v5-artifact-manifest.json`. Deployment provenance is in `kaggle-smoke-v5-deployment-record.json`.

The complete output bundle also contains the notebook's cloned repository; downloaded working output was larger than the requested archive. Only the concise report, log, manifest, and deployment record are committed; checkpoints and the ZIP remain in the local download directory and Kaggle output.

## Acceptance gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Correct private notebook revision and T4×2 metadata | PASS | Version-5 pull matched all 32 staged cell sources; metadata verified. |
| Real cells completed on Kaggle | PASS | Version 5 status `COMPLETE`; captured execution logs. |
| Python 3.13 / NumPy 2 / CUDA Torch compatibility without pip downgrade | PASS | Runtime log; no pip installation in SMOKE. |
| CUDA available and real forward/backward | PASS | CUDA pass in log; two T4 devices identified. |
| CPU REINFORCE and CPU A2C one-update smoke | PASS | Metrics, generation-1 checkpoint and exports for both. |
| CUDA A2C one-update smoke | PASS | CUDA metrics, generation-1 checkpoint and exports. |
| Downloaded manifest and ZIP integrity | PASS | 27 SHA-256 entries, size checks, ZIP CRC and internal entry hashes. |
| No secret exposure | PASS | Authentication reused; secret-pattern scan of captured logs had no matches; no credentials in committed artifacts. |
| EXPERIMENT/FULL_VALIDATION disabled | PASS | Notebook mode gates and captured runtime behavior. |
| Learning acceptance / performance / biological validity | NOT TESTED / NOT CLAIMED | SMOKE only; GPU speedup remains unmeasured. |

This SMOKE result closes only pipeline/runtime correctness gates. The preregistration remains pending/inconclusive, scientific thresholds remain unchanged, historical MVP1 failures remain preserved, and no biological validity is claimed.
