# Pasture Studies: useful-learning autoresearch — final result

**Verdict: FAIL (budget exhausted, 24/24 additional training jobs).** This campaign produced real, loadable exploratory A2C checkpoints and completed real Kaggle T4x2 runs, but no candidate passed the frozen usefulness gate. There is therefore **no accepted/validated checkpoint to promote**. Candidate checkpoints remain available for research inspection only; the local smoke-tested candidates are not evidence of useful learned behavior.

No fifth batch was started. No reserved/locked evaluation data was read. Historical runs and thresholds were preserved. This is a toy simulation result, not a biological validation or husbandry recommendation.

## Scope and frozen gate

The training/evaluation loop used the preregistered mission at `.omx/specs/autoresearch-useful-five-cattle/mission.md`. The gate was not changed: a 3-seed candidate must strictly beat all applicable reference policies on original-reward return, successful forage, and successful water events; final hunger/thirst cannot be worse than best reference; invalid actions must be zero; all common worlds and the full seed cohort count. Candidate training used exploratory variants, but evaluation used the original simulator reward. Evaluation stayed exploratory-only (`formal_eligible=false`, `benchmark_eligible=false`, `locked_test_access=false`).

## Actual Kaggle training

Four Kaggle executions completed with status `COMPLETE`; each trained two independent configurations × three seeds × 512 episodes = six jobs. Thus all 24 jobs completed (12,288 model-episodes total). Each seed advanced 32 optimizer updates. Each execution detected two Tesla T4 GPUs and passed CUDA forward/backward. Independent jobs were scheduled across the devices.

| Batch | Variants / protocol | Kaggle version | Source commit | Training wall time | Two-GPU vs best-single throughput ratio |
|---|---|---:|---|---:|---:|
| 1 | G/H, path-aware resource-rich; original reward | 15 | `4adfdd0ffcc97159868e73f02341634208a4521e` | 356.41 s | 1.857× |
| 2 | I/J, intent-shaping (diagnostic; subsequently found to pay without successful transition) | 17 | `4adfdd0ffcc97159868e73f02341634208a4521e` | 346.68 s | 1.840× |
| 3 | K/L, outcome-aware reward v1 | 19 | `4ed0c9e478e66aadc0111da80a9bee413ed4040c` | 442.22 s | 1.823× |
| 4 | M/N, outcome-aware scale 2.0 | 21 | `3372ed7b1452a0fb92165b15c33092f6049625b2` | 452.10 s | 1.888× |

Kaggle runtime for the runs: Python 3.13.15, NumPy 2.1.3, PyTorch 2.11.0+cu128, CUDA 12.8, 2×Tesla T4. A separate profile found a single GPU was not faster than CPU for the tiny single-model batch: CPU 3.452 s vs CUDA:0 3.635 s for the 16-episode profile. The measured advantage was from independent parallel jobs across two GPUs, not a claim that an individual model trains faster on GPU.

## Locked evaluation result

The final comparison used a new shared evaluation schedule rooted at `9487999`: 4 scenario families (`starter`, `resource_scarce_relocated`, `wide`, `unseen_corridor`), 3 episodes per family, 72 ticks each, five cattle, all three training seeds per candidate, and common trained/untrained/heuristic references. Final evaluation JSON SHA-256 (uncompressed): `1461126a5c0365466e3252bbfe8bc6be29810198848cadc3b60eea65fd4f6e5d`.

| Candidate | Original reward / episode | Forage successes / episode | Water successes / episode | Final hunger | Final thirst | Most frequent actions (mean share) | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| G | 5.397 | 0.028 | 0.472 | 1.000 | 1.000 | REST .696, HERD .204 | FAIL |
| H | 6.424 | 0.028 | 0.417 | 1.000 | 1.000 | REST .705, HERD .227 | FAIL |
| I | -17.915 | 2.222 | 0.528 | .973 | .998 | GRAZE .443, HERD .252 | FAIL |
| J | -16.251 | 2.139 | 0.778 | .996 | .996 | GRAZE .386, HERD .313 | FAIL |
| K | 10.066 | 1.278 | 0.000 | 1.000 | 1.000 | HERD .503, REST .447 | FAIL |
| L | 10.364 | 1.111 | 0.000 | 1.000 | 1.000 | HERD .506, REST .453 | FAIL |
| M | 1.571 | 0.000 | 0.222 | 1.000 | .999 | HERD .509, REST .243, WALK .227 | FAIL |
| N | 2.140 | 0.000 | 0.194 | 1.000 | .999 | HERD .510, REST .252, WALK .220 | FAIL |

Relevant references on those same worlds: the highest reference return was 3.940 (trained B); the best forage rate was 2.500/episode (heuristic); the best water rate was .917/episode (heuristic); lowest reference hunger was .942 (trained E/F); best reference thirst was 1.000. Invalid actions were zero for the candidates, but that alone is not useful learning. No candidate simultaneously reached even the resource-event reference values and none passed the return/hunger/thirst conjunction. I/J improved forage relative to most neural references but still fell below the heuristic forage and water references and had strongly negative common return. J was closest on water among the new cohort (.778 vs .917) but still failed the frozen gate. G/H retained REST collapse; K/L returned strongly through HERD/REST patterns without water-seeking; M/N again favored HERD/WALK and had zero forage.

The I/J outcome was also a reward-design diagnosis: the intent-shaped reward credited resource-seeking actions without requiring a successful physiological transition. K/L removed that exploitable action-level shaping and used observed post-transition state change, but their policies still failed resource behavior; scaling that signal for M/N did not solve it. Training improvement in the shaped objective was therefore not treated as evidence of improvement in the original task.

## Training traces and diagnostic measurements

Mean environment-return summaries from first to final training update were: G −4.443→−0.742; H −4.443→−0.090; I −7.105→−7.287; J −7.105→−7.065; K −6.693→−1.294; L −6.693→−0.916; M −5.974→−2.597; N −5.974→−2.590. The return traces for K–N include their exploratory training reward variants and are not directly comparable as learning objectives; selection used only the common original-reward evaluation above.

Paired memory/perception interventions changed actions in some runs but inconsistently: memory interventions changed actions in 1–3 of 3 seeds depending on candidate, while perception interventions changed actions in 0–1 of 3 seeds for I–N (G/H: perception 1/3). This is evidence of limited local policy sensitivity, not a robust causal benefit to welfare, resource outcomes, or learning. Navigation outcomes were reason-coded; most no-target outcomes for I/J were resource-target availability/competition, while M/N's 458/383 `all_target_cells_occupied` cases reflect dynamic cattle contention. These counts were not silently relabeled as static topology failures. The separate grid-graph regression tests passed.

## Checkpoints and artifact integrity

All four Kaggle artifact ZIPs passed ZIP integrity and SHA-256 verification of every file against their embedded manifest: 433 manifested files per batch, **1,732 file-hash checks total**, and 24 candidate inference checkpoints indexed. All checkpoints are inside the batch ZIPs and their hashes are recorded in `evidence-index.json`; each ZIP contains training metrics, generation/checkpoint data, inference exports and manifests. The two illustrative local simulation smoke outputs are `I-local-simulation-smoke.json` and `J-local-simulation-smoke.json`. They demonstrate that the exports load and run for five animals over 72 ticks; they do **not** validate the behavioral gate. No one checkpoint is promoted or called “best validated.”

The final evaluation JSON is retained gzip-compressed with the original uncompressed SHA recorded in `evidence-index.json`; the gzip contents were round-trip checked. This keeps repository evidence manageable without changing any evaluation bytes. Batch ZIPs and raw Kaggle logs/attestations are retained.

Two illustrative non-accepted exports (candidate I seed 9484104 and candidate J seed 9484105) are directly available under `nonaccepted-candidates/` with matching sidecars. Both loaded and ran in the local simulator smoke. They are supplied only for inspection/reproduction; **neither passed and neither is a validated best checkpoint**.

## Verification performed

- Kaggle jobs: 24 started / 24 completed; all four notebook runs reported `COMPLETE`.
- CUDA: real forward/backward passed in all four runs; two Tesla T4s detected each run.
- Artifact verification: 1,732/1,732 manifest file hashes, plus valid ZIP CRC checks.
- Fresh local regression suite (CPU environment, no local model training): 172 passed, 49 skipped, 1 deselected; 144 subtests passed. The single deselection requires optional PyTorch CPU collector parity and was not run because this mission prohibited local model training. The release-staging cleanliness test passed after the evidence was committed. Focused changed-path tests from the implementation pass: 68 passed, 2 skipped, 1 deselected.
- Python source compile passed. Markdown whitespace checks passed; `git diff --check` reports only the trailing blank terminal lines retained in the four captured Kaggle log JSONs (the original run evidence), not source/report whitespace.
- Full training policy comparison was carried out on Kaggle only. Reserved or formal analysis was not invoked.

## Reproduce / inspect

From the `exploratory-a2c` branch, inspect `validation/autoresearch-useful-five-cattle/evidence-index.json`. Verify a stored archive without executing training:

```bash
python - <<'PY'
import hashlib, json, zipfile
from pathlib import Path
root = Path('validation/autoresearch-useful-five-cattle')
index = json.loads((root / 'evidence-index.json').read_text())
for batch in index['batches']:
    path = root / f"batch-{batch['batch']}" / 'pasture-artifacts.zip'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == batch['archive_sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
    print(batch['batch'], batch['kernel_version'], batch['source_commit'], batch['archive_sha256'])
PY
```

To inspect a candidate export, extract its exact `path_in_archive` from the ZIP listed in the index. Do not treat that file as an accepted policy: the predeclared evaluator returned FAIL for all G–N cohorts. The deployed notebook is `https://www.kaggle.com/code/pascalcoimbra/pasture-studies`.

## Remaining barrier / conclusion

The engineering pipeline completed real training, made outcome-based reward changes in separately versioned exploratory protocols, ran shared-world comparisons, and preserved all checkpoints. The scientific/behavioral goal was **not achieved** within the authorized 24-run budget. Highest value next diagnostic (not run here) is to redesign task observability/action affordances around resource reachability and quantify per-cattle competition while using original-reward held-out evaluation; current evidence does not justify promoting a checkpoint or making a biological claim. Further training requires a new explicit run budget and must retain the failed results and frozen acceptance gates.

The frozen candidate validator was run individually for G–N against the final common evaluation and reference conditions: **8/8 returned `status=failed`**. Their exact JSON outputs, failed-gate booleans, and SHA-256 hashes are under `evaluations/final-evaluation/validators/` and indexed in `evidence-index.json`.
