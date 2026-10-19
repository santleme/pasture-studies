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

## Execution boundary

Full registered evaluations and extensive benchmarks are Kaggle-only. The
commands below are not a supported local workflow: do not run the 30-world
registered evaluation or aggregate benchmark on a workstation. The current
Kaggle launcher is fail-closed for EXPERIMENT/FULL_VALIDATION pending
independently trusted release approval, so those extensive runs are not
currently authorized merely because Kaggle is mandatory.

Local execution may use only an explicitly bounded smoke check. For example,
a small unregistered evaluator smoke must specify `--eval-seed` and a small
`--episodes-per-family`; its output is diagnostic only and is not accepted by
the aggregate report entry point or protocol evidence. Development and
lightweight unit tests remain local. See [Kaggle training](kaggle-training.md)
for the release and execution boundary.

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
