# Pasture Studies datasets and policy benchmarks

This workflow generates offline, reproducible **training-only** artifacts. It
does not establish biological validity. The fixed-corpus compute workload is
stale/off-policy data and is not evidence of policy quality.

Full or mass trajectory generation and extensive benchmarks are Kaggle-only; do
not use the local commands in this document for full research workloads. Local
execution is limited to development, lightweight tests, and explicitly bounded
smoke diagnostics. The current Kaggle release gate still blocks EXPERIMENT and
FULL_VALIDATION absent independent trusted approval.

## Generate and package a dataset

The full PyTorch fixed-corpus workload remains available separately; the dataset
release command below builds three local artifact families using the frozen
training seed split. It defaults to one world per training root (use
`--episodes-per-root` to select a bounded number of preregistered training
worlds):

```bash
python -m validation.pasture_studies_dataset build \
  --output /tmp/pasture-kaggle-dataset --data-version v1 \
  --dataset-id <owner>/<slug> --title "Pasture Studies training data" \
  --license <license-you-have-rights-to-apply>
python -m validation.pasture_studies_dataset verify --bundle /tmp/pasture-kaggle-dataset
```

The package contains `procedural-worlds.jsonl`, `oracle-trajectories.jsonl`,
`dagger-trajectories.jsonl`, Kaggle metadata, and a versioned checksum manifest.
World snapshots use deterministic `farm_sim.make_world`; route rows use the
independent deterministic BFS navigation oracle; DAgger-style rows collect
states visited by a seeded initialized policy, query the existing heuristic
teacher, and retrain the learner on accumulated labels across two deterministic
aggregation rounds. The BFS expert is only an oracle for shortest paths, not
reward-optimal behavior; DAgger labels use the heuristic teacher, not a
biological or optimal expert.
Packaging requires an explicit Kaggle `owner/slug`, title and license. The split
is training-only; held-out validation, pilot, locked-test and reserve worlds are
not packaged. Keep each output directory empty and review its files before any
separate upload step. Kaggle's
official [dataset CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)
creates a dataset from a directory and defaults it to private; publication
requires an explicit `--public`. Version creation uploads directory content as
a new version. The [metadata reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md)
requires the dataset id and a license entry. This repository's packaging command
does not invoke Kaggle, select a license on your behalf, publish, or delete old
versions.

The publisher wrapper validates the package then prints the exact official CLI
command by default. Create and version use Kaggle's documented
`datasets create -p <dir>` and `datasets version -p <dir> -m <notes>` forms:

```bash
python -m scripts.kaggle_dataset_publish create --directory /tmp/pasture-kaggle-dataset
python -m scripts.kaggle_dataset_publish version --directory /tmp/pasture-kaggle-dataset --message "v2 data refresh"
```

Those are local dry-runs; they do not upload. The wrapper intentionally omits
`--public` and old-version deletion flags. A real upload requires the additional
`--publish` opt-in, which is outside this task and was not run.
