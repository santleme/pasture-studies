# Architecture and experiment design

## Runtime flow and boundaries

One `Simulation.step()` advances one integer tick. Time is intentionally abstract: `TICKS_PER_VIRTUAL_DAY = 24`; a committed tick count `t` denotes the next interval `[t, t+1)`, with synthetic day index `t // 24` and within-day tick `t % 24`. Tick 0 is day 0/tick 0; the first step commits interval `[0,1)` and yields world tick 1; yields tick 24 are day 1/tick 0. Every action lasts one abstract tick, with no real-hours mapping. The step takes a common pre-transition snapshot, then processes the five animals in stable ID order for navigation arbitration. Each agent perceives the common snapshot, retrieves relevant events from its own memory, and encodes a fixed feature vector. Policy inference returns an action and six-action probability distribution without mutating state. Navigation resolves intent to at most one legal grid step; physiology and reward are each updated exactly once per animal/tick; memory/events are recorded, then the world and clock commit together. Browser clock speed controls wall-clock scheduling only and cannot alter this logical transition sequence.

```text
FarmWorld + AnimalState
  -> perception.perceive
  -> memory.retrieve (that animal's event history)
  -> perception.encode_features (cattle-observation-v1)
  -> shared policy.decide (pure inference)
  -> navigation.navigate (position only)
  -> physiology.step_physiology (toy drives, reward, labels)
  -> Simulation commits world + ordered events
  -> server JSON API -> browser canvas, inspector, clock and log
```

| Module | Responsibility / contract |
| --- | --- |
| `farm_sim/model.py` | Immutable state records, actions, observation/memory contracts, and the versioned feature-name list. |
| `farm_sim/perception.py` | Bounded grid observation and fixed-order feature encoding. |
| `farm_sim/memory.py` | Per-animal bounded event history and deterministic recent-event retrieval. |
| `farm_sim/policy.py` | NumPy tanh MLP inference, schema-checked checkpoint I/O, and explicit opt-in heuristic comparison baseline. |
| `farm_sim/training.py` | Seeded reward-based training task, evaluation and report generation. |
| `farm_sim/physiology.py` | Pure synthetic drive/reward transition; it does not move animals. |
| `farm_sim/navigation.py` | Deterministic grid path step; it does not update physiology. |
| `farm_sim/simulation.py` | Five-agent scenario creation, tick orchestration, arbitration and event recording. |
| `farm_sim/server.py` | Loopback-only HTTP/API and static file serving; loads a local checkpoint, never a rule fallback. |
| `farm_sim/static/` | Canvas map, clock controls, animal selector/inspector, and decision log. |

## Agents, policy and feature schema

The simulation has **five separate agent states** (`cow-1` through `cow-5`): each has its own position, toy physiology, memory, current action, last reward, last probabilities, and last input features. In contrast, all five use **one shared policy instance and the same checkpoint weights**. The report's five checkpoints are independent training runs, not per-animal models; the UI's default model is the seed-11 checkpoint after the complete training suite passes. Event records are retained in a 5,000-entry ring buffer; this caps process memory and event-response size, but older events are not persisted. IDs increase monotonically within a run and restart at 0 on reset, which also clears the retained history.

The model is a one-hidden-layer NumPy MLP: 20 float inputs, 32 tanh units, and six categorical outputs. Inference is local and selects the highest-probability action. `cattle-observation-v1` fixes the input order:

1. State drives: `hunger`, `thirst`, `fatigue`, `social_need`.
2. Perception/environment: `forage_visible`, `water_visible`, `forage_distance_norm`, `water_distance_norm`, `forage_available`, `water_available`, `local_blocked_fraction`, `herd_count_norm`.
3. Retrieved memory: `mem_water_success`, `mem_forage_success`, `mem_water_blocked`, `mem_forage_depleted`, `mem_event_recency`.
4. Context: `tick_phase_sin`, `tick_phase_cos`, `self_in_herd`.

Names and order are declared in `farm_sim/model.py`; `encode_features` in `farm_sim/perception.py` produces the corresponding vector. Distances and counts are normalized for this toy grid. Memory stores up to 32 events per animal and retrieval considers events within the latest 12 ticks, returning explicit task-relevant markers and up to four event records.

## Training task, reward, and checkpoint validation

The task and historical checkpoint results below describe the pipeline, not permission to execute it locally. Full training is Kaggle-only; local execution is limited to development, lightweight tests, and explicit smoke diagnostics as described in [Kaggle training](kaggle-training.md).

`python3 -m farm_sim.training` runs episodic REINFORCE on a fixed one-step, four-case synthetic task. In two paired cases the animal has visible nearby water and high thirst, or visible nearby forage and low hunger. Exactly one retrieved-memory bit differs within each pair: prior successful access maps to `drink`/`graze`, while blocked/depleted memory maps to `walk`. The target action earns **+1**, every other action **−0.25**. These labels and rewards are invented task semantics, not claims about cattle.

Five independent seeds (`11`, `29`, `47`, `83`, `101`) each train for 2,000 balanced episodes. The REINFORCE update uses a per-case baseline equal to the current policy's expected reward for that case, learning rate 0.07, gradient norm clipping, and seeded sampling. Evaluation uses a separate matched-random-draw stream (`100000 + training seed`): five draws per each of four cases for each run, 100 cases overall. A run suite passes only when aggregate trained accuracy is at least 85%, improvement is at least 25 percentage points, and every checkpoint passes both deterministic paired-memory tests. Those tests require the expected argmax action swap and at least 0.20 contrast in the desired-action probability for both pairs.

`artifacts/policies/training-report.json` records seeds, episode counts, pre/post metrics, per-case results, parameter deltas, probabilities, contrasts, and pass status. The report's observed aggregate was **0.14 initial accuracy**, **0.97 trained accuracy**, **83 percentage-point improvement**, with all five checkpoint counterfactual checks passing. This validates performance on the fixed synthetic task only.

Each checkpoint consists of an `.npz` weight file and a `.json` metadata sidecar. Metadata includes checkpoint version, exact feature/action schema, dimensions, seed, episode count, policy ID, and training metadata. `load_checkpoint` validates schema, dimensions, finite weights, and presence of both files. Saves use temporary files followed by atomic replacement. Training writes `cattle-policy-seed-*.npz` and corresponding `.initial.npz` files, sidecars, and a report. `default-policy.npz` and its sidecar are the last-known-good promoted checkpoint: the seed-11 policy is copied there only after the complete learning and counterfactual gates pass. If a later run completes but fails those gates, its report records the failed criteria and the training command exits unsuccessfully; the existing promoted default and sidecar remain unchanged. Per-seed run outputs and the report may still be refreshed by that run. The explicit `HeuristicBaseline` is an experimental comparison, never an automatic fallback.

## Reproducibility and APIs

The world constructor and training task are seeded. Agents share the same pre-transition snapshot for perception, and stable ID ordering controls movement/resource arbitration. A step increments logical time once; server speed changes only wall-clock pacing. Tests replay identical seeded runs and compare world state and ordered events.

The standard-library HTTP server binds only to `127.0.0.1` (default port 8000):

| Request | Purpose |
| --- | --- |
| `GET /api/state` | Full serialized world, five inspectable agent views, clock state, and policy ID. |
| `GET /api/animals/{id}` | One animal's toy state, perception, memory, current features, decision-time features/probabilities/tick, and recent retained events. Decision probabilities correspond to `decision_features`; current features are a fresh post-transition observation. |
| `GET /api/events?after={id}` | Ordered retained event records after the reset-scoped event ID (`-1` returns all retained records), plus retention metadata. When `has_gap_after_cursor` is true, one or more events before the returned window have already been evicted; clients should refresh from current state rather than assume a complete log. |
| `POST /api/reset` with `{"seed": 7}` | Reset the farm to a deterministic initial scenario. |
| `POST /api/step` with `{}` | Advance one tick. |
| `POST /api/control` with `{"action":"pause"}` or `{"action":"resume"}` | Pause or resume the automatic clock. |
| `POST /api/control` with `{"action":"speed","speed":2}` | Set pacing (0.1–20 ticks per second); does not alter logical `dt`. |

The browser displays the map/resources/obstacles, five selectable cattle, synthetic virtual day and within-day tick (plus clock state), reset/step/run/speed controls, toy drive bars, model probabilities/checkpoint, perception and retrieved memory details, and an append-ordered decision/event log. Event history is bounded to 5,000 records; API retention metadata discloses eviction and resets restart event IDs. No external assets or hosted inference are loaded.

## Validation and limits

The **21/21** test result, **97% vs 14%** task accuracy, and all-counterfactuals-pass status above are historical results from the original fixed-task baseline, not the current remediation status. The current **59-test** full suite passes, including clock speed-invariance coverage and paired three-day replays across 100 seeds using the shipped default policy. Local Playwright/Chromium visual validation is available and passes. Current locked evaluation found the encoder-generated policy collapsed to `HERD_INTERACT` in all 216,000 recorded decisions per family (648,000 total across three families); the four-case policy remained almost entirely `WALK`. The memory and perception influence gates failed, while utility remains inconclusive under the preregistered precision requirement. The locked evaluation's pre-lock attestation did not bind runtime simulation/policy modules; see the [MVP 1 remediation report](../validation/MVP1_REMEDIATION_REPORT.md) for detailed evidence, hashes, and current gate results.

All physiology, memory semantics, reward, and cognition are simplified synthetic software variables. The reward task and 20-feature encoding have no cattle-data calibration or biological validation. This MVP makes no claim of biological accuracy, empirically realistic cattle behavior, animal welfare assessment, or husbandry utility.
