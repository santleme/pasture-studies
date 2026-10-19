# Neural-policy benchmark protocol

This protocol standardizes **descriptive synthetic-simulator** comparisons. It is
not a biological or husbandry validation, and its summaries are not statistical
acceptance tests.

## Protocol and data separation

`validation/preregister_torch_mvp1.py` defines the fixed training, validation,
precision-pilot, locked-test, and reserve seed namespaces. The four evaluation
families in `validation/evaluate_torch_mvp1.py` are `starter`,
`resource_scarce_relocated`, `wide`, and `unseen_corridor`; the last three are
reported as novel-world stress families relative to `starter`. Registered
validation seeds come from the manifest rather than being generated at run
time. The evaluator verifies the manifest binding and reports a passing
train/evaluation disjoint attestation with zero collisions.

Each selected checkpoint is compared on the **same family/world seed** against
an untrained same-seed actor and the explicitly labeled `HeuristicBaseline`.
The raw report retains episode-level reward trajectories, action counts and
frequencies, policy entropy, invalid-action counts, reward-component labels,
navigation reasons, ending drives, and resource state. These are behavior and
simulator-reward measures, not animal outcomes.

Validation is the normal selection/reporting split. The single aggregate report
entry point, `validation/benchmark_policy_report.py`, accepts only schema-v1
reports with paired-world attestations, an explicitly preregistered
`validation` split, and a passing train/evaluation disjoint check with zero
collisions. Locked-test evaluation has a separate authorization and audit
boundary in the evaluator; do not use validation summaries to unlock, inspect,
or make claims from locked outcomes. The candidate protocol also remains
subject to its independently reviewed readiness gates. No result here
authorizes a Kaggle run or publication.

## Local workflow

Install the optional training dependencies in an isolated environment, then
evaluate one committed generation against the registered validation seeds:

```sh
.venv/bin/python validation/evaluate_torch_mvp1.py \
  --generation artifacts/checkpoints/a2c/seed-71001/generation-00000001 \
  --algorithm a2c --seed 71001 \
  --seed-manifest validation/kaggle-mvp1/preregistration.json \
  --split validation --output artifacts/validation/a2c-71001-validation.json

.venv/bin/python validation/benchmark_policy_report.py \
  --input artifacts/validation/a2c-71001-validation.json \
  --output artifacts/validation/a2c-71001-benchmark.json
```

The first command runs 30 registered validation worlds per scenario family and
three policy arms per world. The second validates the report's schema, paired
worlds, validation split, and zero-collision attestation before creating
per-family reward, invalid-action, resource-event, and action-frequency
aggregates; matched trained/untrained-versus-heuristic reward deltas; and
descriptive per-policy generalization gaps from `starter`. It retains raw paired
reward deltas and the preregistration manifest hash. Keep the raw episode report
alongside the aggregate report.

For a small unregistered smoke check only, the evaluator accepts an explicit
`--eval-seed` and `--episodes-per-family`; such an evaluator output is useful
only as a smoke check and is not accepted by the aggregate report entry point or
protocol evidence.

## Interpretation

Compare reward only within this unchanged synthetic reward function. The raw
episode report includes action frequencies, entropy, invalid-action rate,
reward components, navigation reasons, final drives, and resource state to
expose behavioral collapse or tradeoffs that a scalar return hides. The
aggregate report includes per-family resource event counts, paired reward
deltas, and per-policy gaps from the `starter` family. Families intentionally
differ in their scenario construction, so the gaps are descriptive stress
comparisons, not causal estimates. No threshold in these reports means “good
cattle behavior.”
