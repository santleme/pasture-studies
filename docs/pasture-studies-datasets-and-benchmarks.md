# Pasture Studies datasets and policy benchmarks

This workflow generates an offline, reproducible **training-only** simulator
corpus and summarizes held-out validation reports. It does not establish
biological validity. The fixed-corpus performance benchmark is a stale/off-policy
compute workload; it is not evidence of policy quality.

## Generate and package a dataset

Use the existing corpus generator. A smoke corpus is deliberately too small to
package as a dataset or use as acceptance evidence:

```bash
python -m validation.benchmark_torch_fixed_corpus generate-corpus \
  --seed 71001 --output /tmp/pasture-corpus.npz
python -m validation.package_torch_corpus package \
  --corpus /tmp/pasture-corpus.npz --output /tmp/pasture-kaggle-dataset \
  --dataset-id <owner>/<slug> --title "Pasture Studies training corpus" \
  --license <license-you-have-rights-to-apply> --data-version v1
python -m validation.package_torch_corpus verify --package /tmp/pasture-kaggle-dataset
```

Packaging validates corpus hashes, copies only the compressed arrays and their
sidecar, requires an explicit Kaggle `owner/slug`, title and license, and writes
a checksummed manifest. The dataset split is training-only; held-out validation,
pilot, locked-test and reserve worlds are never packaged. Keep each output
directory empty and review its files before any separate upload step. Kaggle's
official [dataset CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)
creates a dataset from a directory and defaults it to private; publication
requires an explicit `--public`. Version creation uploads directory content as
a new version. The [metadata reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md)
requires the dataset id and a license entry. This repository's packaging command
does not invoke Kaggle, select a license on your behalf, publish, or delete old
versions.
