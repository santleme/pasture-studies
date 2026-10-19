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
time. The evaluator verifies the manifest binding and keeps checkpoint training
worlds disjoint from evaluation worlds.

Each selected checkpoint is compared on the **same family/world seed** against
an untrained same-seed actor and the explicitly labeled `HeuristicBaseline`.
The raw report retains episode-level reward trajectories, action counts and
frequencies, policy entropy, invalid-action counts, reward-component labels,
navigation reasons, ending drives, and resource state. These are behavior and
simulator-reward measures, not animal outcomes.

Validation is the normal selection/reporting split. Locked-test evaluation has
a separate authorization and audit boundary in the evaluator. The descriptive
summary helper explicitly rejects reports labeled `locked_test`; do not use it
to unlock, inspect, or make claims from locked outcomes. The candidate protocol
also remains subject to its independently reviewed readiness gates. No result
here authorizes a Kaggle run or publication.

## Local workflow

Install the optional training dependencies in an isolated environment, then
evaluate one committed generation against the registered validation seeds:

```sh
.venv/bin/python validation/evaluate_torch_mvp1.py \
  --generation artifacts/checkpoints/a2c/seed-71001/generation-00000001 \
  --algorithm a2c --seed 71001 \
  --seed-manifest validation/kaggle-mvp1/preregistration.json \
  --split validation --output artifacts/validation/a2c-71001-validation.json

.venv/bin/python validation/summarize_neural_benchmark.py \
  --input artifacts/validation/a2c-71001-validation.json \
  --output artifacts/validation/a2c-71001-validation-summary.json
```

The first command runs 30 registered validation worlds per scenario family and
three policy arms per world. The second creates deterministic family/policy
aggregates, paired trained-minus-baseline reward deltas, and the descriptive
trained-policy gap between starter and the mean of the three novel families.
The summary preserves the input seed-split and generation-manifest hashes.
Aggregate standard deviations are population SDs; there are no confidence
intervals or significance claims in this report. Keep the raw episode report
alongside every summary.

For a small unregistered smoke check only, the evaluator accepts an explicit
`--eval-seed` and `--episodes-per-family`; such output is labeled
`unregistered_descriptive` by the summary and is not protocol evidence.

## Interpretation

Compare reward only within this unchanged synthetic reward function. Use action
frequencies, entropy, invalid-action rate, reward components, navigation
reasons, final drives, and resource state to expose behavioral collapse or
tradeoffs that a scalar return hides. Paired deltas describe matched synthetic
worlds. The generalization gap is the mean total-reward difference across the
three novel families versus `starter`; families intentionally differ in their
scenario construction, so it is a descriptive stress comparison, not a causal
estimate. No threshold in this report means “good cattle behavior.”
