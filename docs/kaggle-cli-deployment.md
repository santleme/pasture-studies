# Kaggle CLI deployment for Pasture Studies

This is a **publish-without-execution** procedure for the reviewed Kaggle notebook. It is deliberately separate from running its notebook cells. It does not authorize a training run or change the notebook's execution gates. Use only `PREFLIGHT` or `SMOKE` if you later run the notebook deliberately; `EXPERIMENT` and `FULL_VALIDATION` remain disabled.

> **Fail closed on the installed CLI.** This environment has Kaggle CLI 2.2.4, whose `kaggle kernels push --help` does not list `--no-run`. Do not run `kaggle kernels push` with that CLI: without the flag, push may execute the notebook. Upgrade through the package manager that installed Kaggle CLI, then check the actual help output. Continue only when the installed `kernels push --help` explicitly shows `--no-run`. Never fall back to an ordinary push.

## 1. Install or upgrade and verify the CLI

Kaggle's official CLI documentation recommends Python 3.11+ and an isolated installation. Use whichever tool owns the existing installation:

```bash
# New isolated install (choose one)
pipx install kaggle
# or: uv tool install kaggle

# Upgrade an existing isolated install (choose the matching tool)
pipx upgrade kaggle
# or: uv tool upgrade kaggle

# For an intentionally managed Python environment, instead use:
python3 -m pip install --upgrade kaggle
```

Confirm which executable is on `PATH`, then inspect version and the safety-critical option. This is a local help check; it does not contact Kaggle or publish anything.

```bash
command -v kaggle
kaggle --version
kaggle kernels push --help
```

Stop unless the help output for the selected executable lists `--no-run`. Do not infer support from a package version number, another Python environment, or online documentation. An upgrade is not itself proof that the active executable supports the option.

## 2. Authenticate interactively when ready

The official CLI supports OAuth login. Run it yourself in a trusted terminal and complete the browser flow:

```bash
kaggle auth login
```

For a headless terminal, `kaggle auth login --no-launch-browser` prints a URL to open manually. The CLI keeps OAuth credentials in its OAuth credential store; do not assume they are stored in the legacy `~/.kaggle/kaggle.json` file. The CLI may also accept token sources or legacy username/key credentials, but those are distinct authentication paths. Do not paste access tokens, API keys, browser callback URLs, or credential files into this repository, notebook, shell history, issue, or deployment record. Do not commit credentials. This guide does not initiate authentication.

## 3. Prepare and review the upload directory

Use a dedicated directory containing the exact reviewed notebook and the CLI metadata. Do not edit the repository notebook or alter its provenance/execution gates as part of deployment setup. The project helper interface is:

```bash
python3 scripts/kaggle_deploy.py preflight --path /tmp/pasture-kaggle-upload --owner <your-kaggle-username> --mode SMOKE
python3 scripts/kaggle_deploy.py prepare --path /tmp/pasture-kaggle-upload --owner <your-kaggle-username> --mode SMOKE --gpu
```

`preflight` checks local prerequisites and validates authentication through the installed Kaggle CLI without starting OAuth. It must not decide OAuth is unavailable solely because `~/.kaggle/kaggle.json` is absent; OAuth credentials use the CLI credential store rather than that legacy file. `prepare` stages metadata for the named owner and mode; the helper accepts only `PREFLIGHT` or `SMOKE` and rejects `EXPERIMENT`/`FULL_VALIDATION`. `--gpu` requests T4×2 metadata but does not itself allocate or execute a GPU workload. Inspect the resulting files and metadata before any publication. You can also use Kaggle's metadata initializer directly, then review/edit its generated template:

```bash
mkdir -p /tmp/pasture-kaggle-upload
# Copy the already-reviewed notebook into this directory before initializing metadata.
kaggle kernels init -p /tmp/pasture-kaggle-upload
```

The generated `kernel-metadata.json` is a template: review it and set its code file to the notebook filename in that directory. For this private, internet-enabled, T4×2 configuration, verify the metadata has the intended equivalent of:

```json
{
  "id": "<your-kaggle-username>/pasture-studies",
  "title": "Pasture Studies",
  "code_file": "pasture_studies_kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

Keep the generated file's schema consistent with the installed CLI's official metadata reference; do not replace `kaggle kernels init` with assumptions about undocumented fields. `NvidiaTeslaT4` is Kaggle's metadata identifier for the T4×2 GPU shape. Availability depends on Kaggle account/quota and platform allocation; metadata is a request, not proof that hardware was granted. Internet access is enabled because this notebook fetches its pinned public GitHub source. Confirm the immutable source pin and all review checks before publishing.

Before any remote action, inspect the upload directory and metadata locally, verify the notebook content is the reviewed one, verify no credentials or unrelated files are present, and record the repository source commit (`git rev-parse HEAD`). The notebook's caller-supplied revision/content hash values remain unverified claims; they do not attest to the active Kaggle notebook bytes or grant experiment eligibility.

## 4. Publish without running (only after the capability check)

After confirming `--no-run` in the active CLI's help and reviewing the directory, the safe publication form is:

```bash
python3 scripts/kaggle_deploy.py publish --path /tmp/pasture-kaggle-upload
```

The helper probes `kaggle kernels push --help` and fails closed if `--no-run` is absent. Its only permitted push form is `kaggle kernels push -p <path> --no-run`; it must not drop the flag or fall back to ordinary push. If invoking the official CLI manually instead, the equivalent safe command (only after the same help check) is:

```bash
kaggle kernels push -p /tmp/pasture-kaggle-upload --no-run
```

`--no-run` saves a notebook version without executing its cells. Do not remove the flag, substitute `kaggle kernels update`, or run a push from a CLI that lacks this option. If the flag is missing or the command's behavior is uncertain, stop and use Kaggle's web UI Quick Save only after separately confirming that it will save without running; do not attempt to make progress by issuing an ordinary CLI push. Publishing is a remote side effect and is not performed by this guide.

## 5. Inspect status, logs, and outputs

These official commands inspect a kernel/version after it exists. A no-run publication does not create new execution output; status/logs/output may refer to a prior run, so identify the exact kernel/version and do not interpret old artifacts as belonging to the new no-run publication.

```bash
python3 scripts/kaggle_deploy.py status <owner>/<kernel-slug>
python3 scripts/kaggle_deploy.py logs <owner>/<kernel-slug>
python3 scripts/kaggle_deploy.py output <owner>/<kernel-slug> --path ./kaggle-output
```

Equivalent official CLI commands are:

```bash
kaggle kernels status <owner>/<kernel-slug>
kaggle kernels logs <owner>/<kernel-slug>
kaggle kernels files <owner>/<kernel-slug>
kaggle kernels output <owner>/<kernel-slug> -p ./kaggle-output
```

Use a version-qualified reference (`<owner>/<kernel-slug>/<version>`) where supported and retain the returned version/URL with the deployment record. `output` downloads output from a run; it does not establish which notebook bytes were executed. Do not run or start a notebook as part of this retrieval step.

## 6. Verify downloaded artifacts and retain provenance

Keep `artifact_manifest.json` beside the downloaded archive. The manifest's per-file `sha256` values and byte sizes are integrity checks: compare each downloaded artifact to the corresponding manifest entry before using it. For a single file:

```bash
sha256sum ./kaggle-output/<artifact-file>
```

Compare the printed digest and `stat -c '%s' ./kaggle-output/<artifact-file>` byte count with that file's manifest entry. The project verifier reads the manifest's `files[]` entries, rejects missing/duplicate/unexpected paths, and checks both size and SHA-256 before artifacts are consumed:

```bash
python3 scripts/kaggle_deploy.py verify ./kaggle-output/artifact_manifest.json --artifacts ./kaggle-output
```

Do not treat a valid artifact hash as evidence of notebook provenance or a successful scientific validation.

Create and retain a deployment record with at least:

- local source commit SHA and the reviewed upload/notebook filename;
- Kaggle CLI version and the help-based `--no-run` capability check result;
- kernel owner/slug and published version/reference, plus publication timestamp;
- metadata settings (private, internet, requested T4×2) and whether this was a no-run save;
- artifact-manifest SHA-256, downloaded artifact paths, and per-file verification outcome if artifacts were retrieved;
- any status/run version associated with retrieved logs or outputs.

The helper can write the record after a no-run publication:

```bash
python3 scripts/kaggle_deploy.py record ./deployment-record.json --kernel <owner>/<kernel-slug> --source-commit "$(git rev-parse HEAD)" --mode SMOKE --metadata /tmp/pasture-kaggle-upload/kernel-metadata.json
```

Review the record locally before sharing it; do not include credentials.

The record is provenance metadata, not an attestation that Kaggle executed those exact local bytes. In particular, never promote the notebook's unverified caller-provided revision or content-hash strings into trusted provenance. The notebook currently hard-disables `EXPERIMENT` and `FULL_VALIDATION`; deployment, a successful smoke, a GPU allocation, or a hash match cannot override those gates or establish benchmark/biological validity.

## Official references

- [Kaggle CLI installation and authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
- [Kaggle kernels commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Kaggle kernel metadata schema](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
- [Kaggle CLI OAuth reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md)
